# Errors

Command failures and integration errors.

---

## [ERR-20260725-009] ablation-varlen-metadata-type

**Logged**: 2026-07-25T09:40:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
The unified ablation benchmark passed Python metadata lists to `chunk_local_cumsum`, whose direct ACLNN wrapper requires NPU tensors.

### Error
```text
TypeError: cu_seqlens must be a torch NPU tensor, got <class 'list'>.
```

### Context
- Dense benchmark cases passed because metadata was absent.
- Later GDN wrappers intentionally accept Python lists, so replacing the canonical list was not appropriate.

### Suggested Fix
Store both canonical Python lists and NPU int64 tensor views in the benchmark input contract; use tensors only for `chunk_local_cumsum`.

### Resolution
- **Resolved**: 2026-07-25T09:41:00+08:00
- **Notes**: Added `cu_seqlens_tensor` and `chunk_indices_tensor` while preserving list metadata for the remaining stages.

---

## [ERR-20260725-008] scoped-opp-package-replaces-shared-opapi

**Logged**: 2026-07-25T09:23:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
Installing a package built with only `chunk_kkt_solve_tri` replaced the wheel's shared opapi library and temporarily removed symbols required by the two-operator baseline.

### Error
```text
Unable to resolve aclnn symbol aclnnChunkScaledDotKktGetWorkspaceSize.
```

### Context
- The installer printed the replacement warning, but `--quiet` treated it as confirmed.
- Source files and build artifacts were unaffected.
- Combination tests require all participating operators to be included in the same package.

### Suggested Fix
Build and install the complete Phase 2 operator set in one package before running baseline, fused, or GDN-core regression tests. Do not install a scoped single-operator package into the shared wheel OPP root.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase2_remote_build.sh

### Resolution
- **Resolved**: 2026-07-25T10:32:00+08:00
- **Notes**: Built and installed one complete Phase 2 package containing the baseline, fused stage, and GDN core symbols before final acceptance.

---

## [ERR-20260725-007] scp-multiple-files-common-parent

**Logged**: 2026-07-25T09:17:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
An SCP command with sources from different subdirectories copied all three files into the operator root instead of their original `op_host` and `op_kernel` locations.

### Error
```text
The transfer succeeded, but the destination path was one directory too high.
```

### Context
- The original remote source files were not overwritten.
- The three misplaced copies had unique filenames and were verified before cleanup.

### Suggested Fix
Use one SCP command per destination directory, then compare local and remote SHA256 values before building.

### Resolution
- **Resolved**: 2026-07-25T09:18:00+08:00
- **Notes**: Re-synced each file to its exact source directory, removed only the verified misplaced copies, and added a hash check before build.

---

## [ERR-20260725-006] opp-installer-before-conda-activation

**Logged**: 2026-07-25T07:54:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The freshly built A2 OPP installer ran under system Python and could not find the installed `fla_npu` wheel.

### Error
```text
fla_npu is not importable. Install flash-linear-attention-npu wheel first.
```

### Context
- The operator build and source-copy SHA256 verification both succeeded.
- The command activated `chw-py11` after, rather than before, running the installer.

### Suggested Fix
Source the conda profile and activate `chw-py11` before invoking the OPP `.run` installer.

### Resolution
- **Resolved**: 2026-07-25T07:54:00+08:00
- **Notes**: Reordered environment activation ahead of installation.

---

## [ERR-20260725-005] powershell-temp-archive-cleanup-policy

**Logged**: 2026-07-25T07:25:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A Phase 2 sync command was rejected before execution because it combined remote transfer with local temporary-file deletion.

### Error
```text
CreateProcess rejected: blocked by policy
```

### Context
- The command attempted to remove an older temporary archive before creating and uploading a new one.
- No archive, source file, or remote worktree was changed.

### Suggested Fix
Use a unique archive name for each sync and split archive creation, upload, and extraction into separate commands.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase2_remote_build.sh

### Resolution
- **Resolved**: 2026-07-25T07:25:00+08:00
- **Notes**: Subsequent syncs use unique non-destructive temporary paths.

---

## [ERR-20260724-004] nested-shell-ssh-quoting

**Logged**: 2026-07-24T21:05:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
An inline PowerShell-to-SSH-to-Bash build command failed before execution because nested quotes and escaped shell variables were parsed inconsistently.

### Error
`bash: -c: line 1: unexpected EOF while looking for matching quote`

### Context
The command embedded a long remote build pipeline inside a local PowerShell string. No build or file mutation occurred remotely.

### Suggested Fix
Use a checked-in or temporary remote script for multi-step remote commands; keep the SSH invocation to a single script path and arguments.

### Metadata
- Reproducible: yes
- Related Files: fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri/op_kernel/chunk_kkt_solve_tri.cpp

### Resolution
- **Resolved**: 2026-07-24T21:05:00+08:00
- **Notes**: Switched subsequent Phase 2 builds to a fixed remote script.

---

## [ERR-20260725-002] powershell-here-string-crlf-remote-exit

**Logged**: 2026-07-25T01:23:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A PowerShell here-string preserved CRLF on the final remote Bash `exit $rc` line.

### Error
```text
bash: exit: 0\r: numeric argument required
```

### Context
- The targeted A2 kernel build itself completed successfully.
- Only the wrapper command reported failure after the build log showed all targets complete.

### Suggested Fix
For short remote builds, use a simple SSH command without an explicit `exit $rc`, or normalize the here-string to LF before piping it to `bash -s`.

### Resolution
- **Resolved**: 2026-07-25T01:23:00+08:00
- **Notes**: Subsequent package/install commands avoid the CRLF-sensitive exit wrapper.

---

## [ERR-20260725-001] powershell-ssh-nested-quoting

**Logged**: 2026-07-25T00:55:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
Nested quoting in a PowerShell `ssh` command corrupted a compound remote Bash command.

### Error
```text
bash: unexpected EOF while looking for matching quote
```

### Context
- A remote install and test command combined local PowerShell quoting, SSH quoting, and a quoted environment path.
- No install or test process started before the parse failure.

### Suggested Fix
Run remote install and test as separate, single-quoted SSH commands and avoid nested command substitutions.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-25T00:56:00+08:00
- **Notes**: Split the operation into two commands; both then executed normally.

---

## [ERR-20260724-004] powershell-embedded-remote-bash-redirection

**Logged**: 2026-07-24T21:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
PowerShell parsed Bash redirection inside an inline Python command before the script reached the remote host.

### Error
```text
Out-File: Could not find a part of the path 'D:\dev\null'.
SyntaxError: unterminated triple-quoted string literal
```

### Context
- A multiline remote Bash script was embedded in `python -c` inside a PowerShell command.
- PowerShell interpreted `2>/dev/null`, then quoting damage truncated the Python string.
- A follow-up here-string preserved shell quoting but the Windows pipeline code page corrupted a non-ASCII Python regex.

### Suggested Fix
Pass multiline Python through a PowerShell here-string (or a temporary script) so Bash syntax is data rather than local shell syntax. Keep the piped Python source ASCII-only and locate configuration using ASCII anchors.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-24T21:00:00+08:00
- **Notes**: Switched subsequent remote orchestration to a PowerShell here-string piped to Python.

---

## [ERR-20260724-002] chunk-kkt-solve-tiling-include-order

**Logged**: 2026-07-24T19:05:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: config

### Summary
CANN 9.1 `kernel_tiling.h` requires fixed-width integer types to be declared by its includer.

### Error
```text
kernel_tiling.h: error: 'uint32_t' does not name a type
```

### Context
- `chunk_kkt_solve_tri_tiling.h` included the CANN header before `<cstdint>`.
- The issue appeared during host tiling compilation on A2.

### Suggested Fix
Include `<cstdint>` before `kernel_tiling/kernel_tiling.h` in new tiling headers.

### Metadata
- Reproducible: yes
- Related Files: fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri/op_host/chunk_kkt_solve_tri_tiling.h

### Resolution
- **Resolved**: 2026-07-24T19:05:00+08:00
- **Notes**: Added the standard integer header before the CANN tiling header.

---

## [ERR-20260724-001] phase2-a2-wheel-build

**Logged**: 2026-07-24T19:00:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
The Phase 2 A2 source bundle omitted a tracked protobuf patch and failed before operator compilation.

### Error
```text
protobuf-hide_absl_symbols.patch: No such file or directory
```

### Context
- The remote source bundle was created from an incomplete file selection rather than a complete tracked checkout.
- The failure occurred in the common third-party patch step, not in `ChunkKktSolveTri`.

### Suggested Fix
Build remote test bundles from a complete repository archive, then overlay only the intended working-tree changes.

### Metadata
- Reproducible: yes
- Related Files: cmake/third_party/build/modules/patch/protobuf-hide_absl_symbols.patch

### Resolution
- **Resolved**: 2026-07-24T19:00:00+08:00
- **Notes**: Restored the missing tracked patch in the isolated A2 test directory before rebuilding.

---

## [ERR-20260724-001] Windows Python OpenMP duplicate runtime

**Logged**: 2026-07-24T18:30:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
The default Windows Python process aborted before ABI tests because two OpenMP runtimes were loaded.

### Error
`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`

### Context
The ABI test does not require real torch. Running Python with `-S` avoids importing site packages and their native OpenMP runtimes.

### Suggested Fix
Run descriptor-only ABI tests with `python -S -m unittest ...`; do not set `KMP_DUPLICATE_LIB_OK`.

### Metadata
- Reproducible: yes
- Related Files: torch_custom/fla_npu/test/test_gdn_core_fwd_ctypes_abi.py

### Resolution
- **Resolved**: 2026-07-24T18:35:00+08:00
- **Notes**: The ABI suite passes with `python -S` after keeping the fake torch module installed for each test module instance.

---

## [ERR-20260724-002] PowerShell rg parameter parsed as ripgrep flag

**Logged**: 2026-07-24T18:30:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A PowerShell-only `-ErrorAction` argument was placed after `rg` and parsed as ripgrep's `-E` encoding option.

### Error
`rg: error parsing flag -E: unknown encoding: rrorAction`

### Context
PowerShell parameters only apply to PowerShell cmdlets, not native executables.

### Suggested Fix
Guard the path with `Test-Path` before invoking `rg`, or redirect native stderr separately.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-24T18:35:00+08:00
- **Notes**: Subsequent checks separated PowerShell path handling from native commands.

---
## [ERR-20260724-003] incremental-kernel-source-copy-stale

**Logged**: 2026-07-24T20:35:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
The incremental A2 build reused a stale copied AscendC kernel source, so new source edits were not present in the installed wheel.

### Error
```text
source SHA256 != build/binary/.../src kernel SHA256
```

### Context
- Repeated Phase 2 synchronization experiments appeared to have identical runtime behavior.
- `build/binary/ascend910b/src/chunk_kkt_solve_tri/op_kernel/chunk_kkt_solve_tri.cpp` still matched the original kernel.
- `--incremental` did not invalidate the operator source-copy stamp for these edits.

### Suggested Fix
Before trusting an incremental kernel test, compare source and copied-build SHA256 values. Remove the operator's source-copy and compile stamps, or perform a clean operator build when they differ.

### Metadata
- Reproducible: yes
- Related Files: fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri/op_kernel/chunk_kkt_solve_tri.cpp

### Resolution
- **Resolved**: 2026-07-24T20:35:00+08:00
- **Notes**: Added an explicit SHA256 verification step before A2 runtime testing and forced the operator build artifacts to refresh.

---
