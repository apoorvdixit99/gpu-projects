# Issues & Notes

Log of issues encountered during development and the fixes applied.

---

## Compilation

### First-run JIT compilation is slow
`torch.utils.cpp_extension.load()` invokes nvcc and MSVC the first time the
extension is loaded.  On a typical system this takes 30–60 seconds.  Subsequent
runs use the cached binary in `~/.cache/torch_extensions/` and load instantly.

### VS 2022 Build Tools required (VS 2019 does not work)
**Symptom:** `Command '['where', 'cl']' returned non-zero exit status 1`  
**Root cause:** `torch.utils.cpp_extension` calls `where cl` before invoking nvcc.
`cl.exe` (the MSVC compiler) is not on PATH by default — it must be activated via
`vcvarsall.bat`.

**Additional constraint:** PyTorch 2.12 uses C++20 features in its headers
(`torch/headeronly/util/TypeList.h` and others). nvcc 12.6 detects the host
compiler version and silently drops the `-std=c++20` flag when MSVC 2019 is
detected (`nvcc warning: The -std=c++20 flag is not supported with the configured
host compiler. Flag will be ignored.`). This causes PyTorch's C++20 headers to
fail to compile. VS 2019 Build Tools (MSVC 14.29) cannot be used with
PyTorch 2.12+. **VS 2022 Build Tools (MSVC 14.4x) is required.**

**Fix:** Install VS 2022 Build Tools with the "Desktop development with C++"
workload from `https://aka.ms/vs/17/release/vs_BuildTools.exe`.

### VS 2022 Build Tools installs to Program Files (x86), not Program Files
On this machine, VS 2022 Build Tools installed to:
```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\
```
not the expected `C:\Program Files\` path. The correct `vcvarsall.bat` is:
```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat
```

### MSVC environment must be loaded before running Python
`cl.exe` and the MSVC lib/include paths are not on the system PATH by default.
They must be activated for each terminal session by running `vcvarsall.bat`.

**Fix:** Add the following block to the PowerShell profile
(`$PROFILE` = `C:\Users\<user>\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1`)
so it loads automatically on every new terminal:

```powershell
$_vc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if (Test-Path $_vc) {
    $envLines = cmd /c "`"$_vc`" x64 2>nul && set"
    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    Remove-Variable _vc
}
```

### ninja not on PATH
**Symptom:** `RuntimeError: Ninja is required to load C++ extensions`  
**Root cause:** `ninja.exe` is installed inside the venv (`\.venv\Scripts\ninja.exe`)
but the venv Scripts directory is only added to PATH when the venv is activated via
`Activate.ps1`. If the venv is not activated before running Python, ninja is not found.  
**Fix:** Always activate the venv first: `.venv\Scripts\Activate.ps1`

---

## Benchmarking

### CPU baselines skipped above threshold
NumPy and PyTorch-CPU are not benchmarked for N > 64M (vector add / reduction)
or N > 1024 (matmul) because their latency becomes too large for practical
comparison.  The `--no-cpu` flag skips all CPU baselines.

---

## Nsight Tools

*(Add notes here as issues arise during profiling.)*
