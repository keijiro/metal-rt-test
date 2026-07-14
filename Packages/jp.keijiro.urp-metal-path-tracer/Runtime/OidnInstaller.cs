#if UNITY_EDITOR

using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Debug = UnityEngine.Debug;

namespace UrpMetalPathTracer {

// Installs the Open Image Denoise runtime into the project Library folder
// on first use. The path tracer is editor-only, so the OIDN binaries are
// not bundled with the package; the official release archive is downloaded
// once per project instead. The native plugin keeps denoising disabled
// until the library path is handed over, so a failed or in-flight install
// gracefully degrades to non-denoised output.
static class OidnInstaller
{
    const string Version = "2.5.0";
    const string ArchiveUrl =
      "https://github.com/RenderKit/oidn/releases/download/v" + Version +
      "/oidn-" + Version + ".arm64.macos.tar.gz";
    const string ArchiveSha256 =
      "586142ec125de0bf5b01d3cc4c76985d4fafb0fc91e9f6562e32f3b669f86be5";

    static string RootDir => Path.GetFullPath(Path.Combine
      ("Library", "jp.keijiro.urp-metal-path-tracer", "oidn-" + Version));

    static string LibraryPath => Path.Combine
      (RootDir, "lib", "libOpenImageDenoise." + Version + ".dylib");

    static readonly string[] ArchiveMembers =
    {
        "oidn-" + Version + ".arm64.macos/lib/libOpenImageDenoise." +
          Version + ".dylib",
        "oidn-" + Version + ".arm64.macos/lib/libOpenImageDenoise_core." +
          Version + ".dylib",
        "oidn-" + Version +
          ".arm64.macos/lib/libOpenImageDenoise_device_metal." +
          Version + ".dylib"
    };

    static bool _started;

    public static void EnsureInstalled()
    {
        if (_started) return;
        _started = true;

        if (File.Exists(LibraryPath))
        {
            MetalRTPlugin.MetalRT_SetOidnLibraryPath(LibraryPath);
            return;
        }

        Task.Run(Install);
    }

    static async Task Install()
    {
        try
        {
            Debug.Log("[MetalRT] Downloading Open Image Denoise " + Version +
                      " (~50MB); denoising activates when it completes.");

            using var client = new HttpClient();
            var bytes = await client.GetByteArrayAsync(ArchiveUrl);

            using (var sha = SHA256.Create())
            {
                var hash = BitConverter.ToString(sha.ComputeHash(bytes))
                             .Replace("-", "");
                if (!hash.Equals(ArchiveSha256,
                                 StringComparison.OrdinalIgnoreCase))
                    throw new Exception("Archive checksum mismatch: " + hash);
            }

            Directory.CreateDirectory(RootDir);
            var archive = Path.Combine(RootDir, "download.tar.gz");
            File.WriteAllBytes(archive, bytes);
            Extract(archive);
            File.Delete(archive);

            if (!File.Exists(LibraryPath))
                throw new Exception("Extraction did not produce " +
                                    LibraryPath);

            MetalRTPlugin.MetalRT_SetOidnLibraryPath(LibraryPath);
            Debug.Log("[MetalRT] Open Image Denoise installed to " + RootDir);
        }
        catch (Exception e)
        {
            Debug.LogWarning("[MetalRT] Open Image Denoise install failed; " +
                             "denoising disabled: " + e.Message);
        }
    }

    static void Extract(string archive)
    {
        var args = "-xzf \"" + archive + "\" -C \"" + RootDir +
                   "\" --strip-components 1";
        foreach (var member in ArchiveMembers) args += " \"" + member + "\"";

        var info = new ProcessStartInfo("/usr/bin/tar", args)
          { UseShellExecute = false, RedirectStandardError = true };
        using var proc = Process.Start(info);
        var error = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        if (proc.ExitCode != 0)
            throw new Exception("tar failed: " + error.Trim());
    }
}

} // namespace UrpMetalPathTracer

#endif
