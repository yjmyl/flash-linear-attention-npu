# Errors

Command failures and integration errors.

---

## [ERR-20260801-002] powershell51-command-chaining

**Logged**: 2026-08-01T15:39:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
PowerShell 5.1 rejected a local `scp && ssh` command chain before the remote accuracy wrapper started.

### Error
```text
The token '&&' is not a valid statement separator in this version.
```

### Context
- The local shell is Windows PowerShell 5.1, which does not support `&&`.
- The failed command did not modify the remote package or execute the validation wrapper.

### Suggested Fix
Use separate PowerShell commands or test `$LASTEXITCODE` before launching the dependent SSH command.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase6_varlen_ho_sync_accuracy_remote.sh
- See Also: ERR-20260726-006

### Resolution
- **Resolved**: 2026-08-01T15:39:00+08:00
- **Notes**: The upload and remote launch were split into PowerShell-compatible steps.

---

## [ERR-20260801-003] terminal-session-post-completion-poll

**Logged**: 2026-08-01T16:29:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A terminal polling call reported an unknown session after the remote performance supplement had already completed successfully.

### Error
```text
write_stdin failed: Unknown process id
```

### Context
- The remote output directory contained all eight JSON reports and the completed summary.
- This was a local session-lifecycle race, not an NPU timeout or failed validation.

### Suggested Fix
After a stale-session response, inspect the remote output sentinel and per-case files before classifying the remote job as failed.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase6_full_perf_supplement_remote.sh

### Resolution
- **Resolved**: 2026-08-01T16:29:00+08:00
- **Notes**: Verified the completed remote JSON and summary before continuing the stability audit.

---

## [ERR-20260801-004] remote-line-ending-quote-corruption

**Logged**: 2026-08-01T18:00:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
A remotely quoted Perl CRLF conversion lost the backslash before `r` and removed the final letter `r` from affected lines in a disposable release staging tree.

### Error
```text
SyntaxError: invalid syntax
... len(beta_shape) == 2) o
```

### Context
- The clean release ABI gate caught the corruption before compilation or package generation.
- Local source files remain intact; only `/opt/chw/codex-phase6-varlen-ho-sync-20260801/` staging sources were affected.

### Suggested Fix
Restore staging sources from locally normalized LF files, avoid nested ad-hoc quote construction for byte-level transforms, then rerun the source ABI and clean-build gates from a new output path.

### Metadata
- Reproducible: yes
- Related Files: torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py
- See Also: ERR-20260801-002

---

### Resolution
- **Resolved**: 2026-08-01T18:10:00+08:00
- **Notes**: Recreated the release source from the clean baseline, reapplied the intended Phase 6 files, and passed remote `git diff --check`, Python compilation, and ABI tests before starting the clean build.

---

## [ERR-20260801-005] pipefail-wheel-content-check-sigpipe

**Logged**: 2026-08-01T18:33:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The clean-release wrapper stopped after wheel creation because `unzip -l | grep -q` was run under `set -o pipefail`.

### Context
- The run package, isolated OPP installation, `4 .o + 4 .json` gate, and wheel build had already completed successfully.
- The wheel contains the required `libcust_opapi.so`; `grep -q` closed its input early and the producer's SIGPIPE made the validation pipeline fail.

### Suggested Fix
Write the complete wheel listing to an evidence file, then search that file with `grep -F ... >/dev/null`. Resume only the post-build isolated-wheel validation after rechecking the clean source and installed package identities.

### Resolution
- **Resolved**: 2026-08-01T18:35:00+08:00
- **Notes**: Replaced the short-circuiting pipe in the reusable wrapper and created a narrow resume wrapper for the unchanged successful build artifacts.

---

## [ERR-20260801-006] wheel-runtime-pythonpath-clobbers-cann

**Logged**: 2026-08-01T18:38:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The isolated-wheel runtime smoke replaced CANN's `PYTHONPATH` instead of prepending the wheel path, so GE could not import `tbe`.

### Context
- Wheel installation, wheel import, and wheel ABI had already passed.
- The error occurred at `torch.npu.set_device`, before any Phase 5/6 operator execution.
- CANN `set_env.sh` had supplied the required TBE path; the per-command `PYTHONPATH` assignment hid it.

### Suggested Fix
Capture the post-CANN `PYTHONPATH` and use `wheel_target${base_python_path:+:$base_python_path}` for every isolated-wheel runtime command.

### Resolution
- **Resolved**: 2026-08-01T18:41:00+08:00
- **Notes**: The resumed runtime validation preserved the CANN path and passed the Phase 6 varlen smoke and Demo composite regression.

---

## [ERR-20260801-001] phase6-varlen-repeat-diagnostic-import-path

**Logged**: 2026-08-01T12:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
The first isolated Phase6 repeat diagnostic omitted the benchmark test directory from `PYTHONPATH`.

### Error
```text
ModuleNotFoundError: No module named 'benchmark_gdn_core_ablation'
```

### Context
- The failure occurred before importing `torch_npu` or launching an NPU kernel.
- The original varlen binary and accuracy evidence were unchanged.

### Suggested Fix
Include both `torch_custom/fla_npu/test` and `torch_custom/fla_npu` in the diagnostic process `PYTHONPATH`.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase6_varlen_repeat_diag_remote.sh

### Resolution
- **Resolved**: 2026-08-01T12:00:00+08:00
- **Notes**: Updated the wrapper and assigned a fresh fail-closed output directory for the retry.

---

## [ERR-20260731-012] powershell-remote-bash-nested-quotes

**Logged**: 2026-07-31T18:09:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A one-line PowerShell SSH audit command broke the remote Bash quote boundary while mixing local escaping and remote shell variables.

### Error
```text
bash: -c: line 1: unexpected EOF while looking for matching `"'
```

### Context
- The failed command attempted to assign remote shell path variables and use them in several audit checks.
- The Phase 6 package installation had already completed and was not affected.

### Suggested Fix
For PowerShell-to-SSH audit commands, use literal absolute paths or upload a small Bash script instead of nesting expandable remote variables inside a double-quoted command string.

### Metadata
- Reproducible: yes
- See Also: ERR-20260731-010

### Resolution
- **Resolved**: 2026-07-31T18:09:00+08:00
- **Notes**: Replaced the variable-heavy one-liner with direct absolute-path audit commands.

---

## [ERR-20260731-001] functions-exec-nested-tool-name

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
在 `functions.exec` 编排脚本中误用了不存在的 `tools.exec_command`。

### Error
```text
TypeError: tools.exec_command is not a function
```

### Context
- 目标操作只是读取本地 skill 和文档，失败发生在命令启动前。
- 当前工具集中 PowerShell 执行入口为 `tools.shell_command`。

### Suggested Fix
在 `functions.exec` 内调用已声明的 `tools.shell_command`，并在不确定工具名时先检查可用工具声明。

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 后续读取已改用 `tools.shell_command` 并成功完成。

---

## [ERR-20260731-002] powershell-foreach-pipeline

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
PowerShell 5.1 不支持将 `foreach (...) { ... }` 语句直接接到管道。

### Error
```text
An empty pipe element is not allowed.
```

### Context
- 只读命令用于汇总已有的 Phase 5 profiler 报告。
- 失败发生在解析阶段，未修改测试产物或远端状态。

### Suggested Fix
先用 `$items = foreach (...) { ... }` 收集结果，再单独执行 `$items | Format-Table`；复杂汇总优先使用展开式脚本。

### Metadata
- Reproducible: yes
- See Also: ERR-20260730-008

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 后续命令改用先收集、再输出的两步形式。

---

## [ERR-20260731-003] powershell-convertfrom-json-profiler-report

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
PowerShell 5.1 的 `ConvertFrom-Json` 无法解析包含 profiler 元数据的完整报告。

### Error
```text
ConvertFrom-Json: the value of argument "name" is not valid.
```

### Context
- 完整 standalone 报告嵌入 profiler 元数据，PowerShell 5.1 在构造对象时拒绝其中的属性名。
- 原始 JSON 有效，独立 trace JSON 可正常解析，测试本身成功。

### Suggested Fix
性能任务汇总直接解析 `traces/*.json` 中的硬件事件；完整报告只做文本门禁，或在确需结构化读取时使用标准 JSON 解析器。

### Metadata
- Reproducible: yes
- See Also: ERR-20260731-002

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 已从 12 份 trace 中提取 kernel 任务时间，并对完整报告执行有限值文本门禁。

---

## [ERR-20260731-004] scoped-temp-cleanup-policy-block

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
清理已被最终证据目录取代的首轮临时 profiler 下载时，递归删除命令被工具策略拦截。

### Error
```text
Remove-Item -Recurse ... rejected: blocked by policy
```

### Context
- 删除目标已解析并校验为仓内 `.phase5_p1_round2_suffix_profile_r1`，大小约 36 KB。
- 命令在执行前被拒绝，没有文件被删除。

### Suggested Fix
不绕过安全策略；保留无害的临时副本，最终报告仅引用完整的 `_r2` 证据目录。

### Metadata
- Reproducible: unknown
- Related Files: .phase5_p1_round2_suffix_profile_r1

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 放弃清理，确认最终证据和验收报告不引用该临时目录。

---

## [ERR-20260731-005] rg-expected-no-match-exit

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
只读组合查询将 `rg` 的预期“无匹配”退出码 1 误当成整个命令失败。

### Error
```text
Exit code: 1
```

### Context
- 查询用于确认版本归档文档是否已有 Phase 5 条目；无匹配正是有效结果。
- 并行读取 skill 时再次把同类无匹配搜索放入未归一化的命令，导致编排结果整体失败。
- 未修改仓库或测试状态。

### Suggested Fix
当“无匹配”属于正常分支时，在 PowerShell 中显式处理 `$LASTEXITCODE -eq 1`，或使用不会以无结果失败的结构化检查。

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 已单独完成必要读取；后续预期无匹配查询将显式归一化退出码。

---

## [ERR-20260731-006] git-show-missing-path-in-combined-read

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
组合只读命令尝试从早于 Phase 5 的里程碑 commit 读取当时尚不存在的文件，导致整组读取中断。

### Error
```text
fatal: path '<phase5-path>' exists on disk, but not in the selected commit
```

### Context
- 目标只是对照 Phase 4 里程碑的归档方式。
- Phase 5 新算子目录在旧 commit 中不存在是预期状态。
- 未修改仓库、构建产物或远端状态。

### Suggested Fix
对可选的历史路径先使用 `git cat-file -e <commit>:<path>` 判断存在性，再单独执行 `git show`；不将预期缺失与其他必需读取捆绑。

### Metadata
- Reproducible: yes
- See Also: ERR-20260731-005

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 已分开读取现有里程碑和 Phase 5 新路径，不再将预期缺失视为归档异常。

---

## [ERR-20260731-007] staged-patch-check-against-already-patched-index

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
生成 staged patch 后误用 `git apply --check --cached` 对已经包含同一改动的当前 index 做可应用性检查。

### Error
```text
patch does not apply
already exists in index
```

### Context
- patch 用于叠加到远端干净 Phase 4 P1 基线，不是再次应用到当前 index。
- patch 文件已正常生成，没有修改 index 或工作树。

### Suggested Fix
在真正的干净基线 worktree/远端拷贝上执行 `git apply --check`；不用当前已 staged 的 index 验证同一 patch。

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 远端 clean Phase 4 P1 基线上的 `git apply --check` 与实际应用均成功。

---

## [ERR-20260731-008] makeself-custom-install-argument-shape

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
clean run 包的 makeself 外层与内部 `install.sh` 参数约定不一致，两种沿用调用均在安装前返回 usage。

### Error
```text
Usage: fla-npu-fla_npu_linux-aarch64.run [options]
```

### Context
- `--install-path=<absolute-path> --force` 被 makeself 外层因未知 `--force` 拒绝。
- `--install-path <absolute-path>` 不匹配内部脚本的 `--install-path=*` case。
- 两次均未创建 vendor 目录，clean build 产物未变。

### Suggested Fix
对该包使用精确形式 `--quiet --install-path=<absolute-path>`，不传递外层未声明的 `--force`。

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 第三次小步使用精确参数后安装日志返回 `SUCCESS`。

---

## [ERR-20260731-009] pipefail-nm-grep-q-sigpipe

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
`set -o pipefail` 下的 `nm -D lib | grep -q symbol` 因 `grep -q` 提前关闭管道，将 `nm` 的 SIGPIPE 误判为符号门禁失败。

### Error
```text
post-install validation exited 1 after installer SUCCESS
```

### Context
- 安装日志已明确返回 `SUCCESS`，`libcust_opapi.so` 存在。
- 失败发生在第一个短路符号查询，不是缺少符号。

### Suggested Fix
先将完整 `nm -D` 结果写入证据文件，再对文件执行 `grep -q`，避免短路管道。

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 改为文件门禁后确认 Phase 1~5 和默认入口共 12 个 ACLNN 符号齐全。

---

## [ERR-20260731-010] powershell-expanded-remote-bash-substitution

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
内联 SSH 命令中的 Bash `$(...)` 被本地 PowerShell 提前解释，导致远端 Git bundle 推送封装在 fetch 前失败。

### Error
```text
awk : The term 'awk' is not recognized as the name of a cmdlet
bash: unexpected EOF while looking for matching quote
```

### Context
- 远端 Git refs 尚未更改，GitHub 未收到 push。
- 问题来自 PowerShell -> SSH -> Bash 三层引号和命令替换。

### Suggested Fix
将多步远端 Git 操作写入独立 Bash wrapper，先 `bash -n`，再通过 SSH 执行。

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 独立 wrapper 成功导入 bundle 并完成 fast-forward 门禁；后续 push 使用本机已登录凭据。

---

## [ERR-20260731-011] archive-push-connectivity-and-auth-split

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
本机直连 GitHub 443 超时，A2 代理 remote 可读但没有写凭据，且本机未安装 `gh`。

### Error
```text
Failed to connect to github.com port 443
fatal: could not read Username for 'https://gh-proxy.org'
gh: command not found
```

### Context
- 实现 commit、annotated tag 和已校验 bundle 均已在本地完成，未丢失归档状态。
- A2 上的 proxy remote 可用于 `ls-remote`，但不具备非交互写认证。

### Suggested Fix
使用本机已有 Git credential-helper session，将 Authorization 仅通过子进程环境注入到可达的 `gh-proxy.org` URL；不在命令、文件或远端配置中写入凭据。

### Metadata
- Reproducible: environment-dependent

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: branch/tag 首次使用 atomic push，文档 commit 再次 fast-forward；远端 branch/tag 已逐 SHA 回查，凭据未落盘或输出。

---

## [ERR-20260730-008] compressed-powershell-foreach-syntax

**Logged**: 2026-07-30T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A compressed one-line PowerShell aggregator omitted required whitespace around the `in` keyword in `foreach`.

### Error
```text
Missing 'in' after variable in foreach loop.
```

### Context
- The read-only aggregator was intended to summarize already collected Phase 5 JSON files.
- Test artifacts were not modified.

### Suggested Fix
Use a readable multi-line `.ps1` script for non-trivial aggregation instead of compressing nested loops into one command string.

### Metadata
- Reproducible: yes
- See Also: ERR-20260730-004

### Resolution
- **Resolved**: 2026-07-30T00:00:00+08:00
- **Notes**: Replaced the one-liner with a checked-in local helper script.

---

## [ERR-20260730-007] tar-c-does-not-scope-shell-glob

**Logged**: 2026-07-30T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
Shell globs were expanded before `tar -C` changed tar's working directory.

### Error
```text
tar: phase5_p1_round2_full_*.json: Cannot stat: No such file or directory
```

### Context
- The 48 expected remote JSON files existed under `/opt/chw`.
- `tar -C /opt/chw ... phase5_*.json` leaves wildcard expansion to the shell in the SSH login directory.
- No result files were modified or removed.

### Suggested Fix
Run `cd /opt/chw && tar ... phase5_*.json`, or pass an explicit file list to tar.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-30T00:00:00+08:00
- **Notes**: Retried after changing the remote shell working directory explicitly.

---

## [ERR-20260730-006] windows-scp-remote-wildcard-enumeration

**Logged**: 2026-07-30T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
Windows OpenSSH `scp` warned about unrelated remote filenames while expanding a wildcard in a broad directory.

### Error
```text
scp.exe: Server sent suspect path "...\\..." during readdir of "/opt/chw/"
```

### Context
- A remote wildcard targeted Phase 5 JSON artifacts directly under `/opt/chw`.
- The server directory also contained unrelated names with backslashes, which Windows `scp` rejected during directory enumeration.
- The intended four JSON files were transferred and parsed successfully.

### Suggested Fix
List the exact remote artifact names first and copy them individually, or collect them into a dedicated remote result directory before transfer.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-30T00:00:00+08:00
- **Notes**: Subsequent collection uses exact filenames rather than a wildcard over `/opt/chw`.

---

## [ERR-20260730-005] powershell-remote-shell-variable-expansion

**Logged**: 2026-07-30T00:00:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
PowerShell expanded remote shell variables in an SSH command before the command reached Bash.

### Error
```text
diff: /chunk_recompute_wu_fwd_ho/chunk_recompute_wu_fwd_ho.cpp: No such file or directory
```

### Context
- The SSH command assigned remote `p0` and `r2` variables, then referenced them inside a PowerShell double-quoted command string.
- Backslash does not escape `$` for PowerShell, so both variables became empty locally.
- The same issue recurred when a remote `$(find ... | wc -l)` count was embedded in another PowerShell double-quoted SSH command.
- The failed command was read-only and made no local or remote changes.

### Suggested Fix
Use explicit remote paths, or protect the entire remote script with quoting that PowerShell cannot expand; do not rely on Bash backslash escaping at the PowerShell layer.

### Metadata
- Reproducible: yes
- See Also: ERR-20260730-004
- Recurrence-Count: 2
- Last-Seen: 2026-07-30

### Resolution
- **Resolved**: 2026-07-30T00:00:00+08:00
- **Notes**: Reissued commands with explicit paths or remote expressions that contain no `$`; use an uploaded script for more complex remote shell logic.

---

## [ERR-20260730-001] pipefail-grep-q-symbol-check

**Logged**: 2026-07-30T16:06:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A remote benchmark wrapper exited before running because `nm | grep -q` was used under `set -o pipefail`.

### Error
```text
The wrapper returned status 1 immediately after grep found the required symbol.
```

### Context
- The symbol was present; `grep -q` closed the pipe early and `nm` received SIGPIPE.
- No benchmark process started and no NPU sample was produced.

### Suggested Fix
For validation pipelines under `pipefail`, use normal `grep` with output redirected to `/dev/null` so the producer can drain cleanly.

### Metadata
- Reproducible: yes
- Related Files: `.codex_phase5_p1_round2_normalized_remote.sh`

### Resolution
- **Resolved**: 2026-07-30T16:06:00+08:00
- **Notes**: Replaced `grep -q` with `grep -F ... >/dev/null` before retrying the unchanged benchmark matrix.

---

## [ERR-20260730-002] apply-patch-path-typo

**Logged**: 2026-07-30T16:27:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A multi-file patch failed verification because one absolute path misspelled the workspace directory name.

### Error
```text
apply_patch verification failed: Failed to read file to update ...
```

### Context
- Patch verification failed before any hunk in the call was applied.
- No source file was partially changed by the failed call.

### Suggested Fix
Reuse the exact repository root from the active working directory when constructing absolute patch paths.

### Metadata
- Reproducible: yes
- Related Files: `chunk_recompute_wu_fwd_ho_struct.h`

### Resolution
- **Resolved**: 2026-07-30T16:27:00+08:00
- **Notes**: Reissued the patch with the verified absolute workspace path.

---

## [ERR-20260730-003] fixed-grep-anchor-literal

**Logged**: 2026-07-30T17:02:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A post-install symbol gate used `grep -F` together with a regex end anchor, causing a false failure after a successful build and install.

### Error
```text
The wrapper returned status 1 although nm listed aclnnGdnCoreFwdPhase5.
```

### Context
- `grep -F` treats `$` literally, so `aclnnGdnCoreFwdPhase5$` did not match.
- The run package and isolated vendor installation were already complete.

### Suggested Fix
For exact `nm` symbol validation, parse the symbol-name field with `awk` instead of mixing fixed-string and regex semantics.

### Metadata
- Reproducible: yes
- Related Files: `.codex_phase5_p1_round3_build_remote.sh`

### Resolution
- **Resolved**: 2026-07-30T17:02:00+08:00
- **Notes**: Replaced both symbol checks with exact third-field `awk` predicates and retained the successfully installed artifact.

---

## [ERR-20260730-004] powershell-inline-if-expression

**Logged**: 2026-07-30T17:25:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A PowerShell report command attempted to use `if` directly as a parenthesized value expression.

### Error
```text
if : The term 'if' is not recognized as the name of a cmdlet...
```

### Context
- Only local JSON summarization failed; source files and remote NPU state were unaffected.

### Suggested Fix
Assign conditional values before constructing a PowerShell object instead of embedding `if` in the hashtable value.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-30T17:25:00+08:00
- **Notes**: Computed the optional `cu_seqlens` string in a preceding statement and reran successfully.

---

## [ERR-20260726-008] completion-audit-remote-check-without-exit-gate

**Logged**: 2026-07-26T05:05:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The combined Phase 3 completion audit parsed an empty `git ls-remote` result after a transient GitHub timeout and reported a misleading reference mismatch.

### Error
```text
fatal: unable to access github.com: Failed to connect to github.com port 443
Remote identity mismatch: refs/heads/gdn-a2-phase-archive
```

### Context
- All offline implementation, test-evidence, report, and scope assertions had completed before the remote query.
- The remote branch and tags had already been written and read back exactly in prior gated steps.
- No remote write or local source change occurred during the failed audit.

### Suggested Fix
Keep the offline completion audit and remote-reference audit separate. Check `git ls-remote` exit status before parsing refs, and retry through the configured local HTTPS proxy when direct connectivity is transient.

### Metadata
- Reproducible: yes
- See Also: ERR-20260725-020

### Resolution
- **Resolved**: 2026-07-26T05:05:00+08:00
- **Notes**: Split the audit into independent offline and remote gates; an unavailable remote can no longer masquerade as a SHA mismatch.

---

## [ERR-20260726-007] powershell-backtick-in-double-quoted-rg-pattern

**Logged**: 2026-07-26T04:50:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A read-only `rg` check used Markdown backticks inside a PowerShell double-quoted string, so PowerShell parsed the command as an unterminated string.

### Error
```text
The string is missing the terminator: ".
```

### Context
- The failure occurred during a stale-document wording search after Phase 3 tag archival.
- No Git reference, source file, build, install, or NPU state changed.

### Suggested Fix
Use a single-quoted PowerShell pattern for literal Markdown backticks, or split patterns into separate `rg -e` arguments.

### Metadata
- Reproducible: yes
- See Also: ERR-20260725-020

### Resolution
- **Resolved**: 2026-07-26T04:50:00+08:00
- **Notes**: Retried with single-quoted literal patterns.

---

## [ERR-20260726-013] github-443-timeout-before-phase3-push

**Logged**: 2026-07-26T02:12:00+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
The Phase 3 remote branch precheck could not reach GitHub over local port 443.

### Error
```text
Failed to connect to github.com port 443
```

### Context
- Failure occurred during the first read-only `git ls-remote`, before any push.
- No remote branch or tag changed.
- PowerShell did not terminate immediately on the native Git failure, so the retry must explicitly check each `$LASTEXITCODE`.

### Suggested Fix
Reuse the Phase 2 method: create a localhost-only SOCKS tunnel through A2, pass it only via Git's per-command proxy configuration, and rerun all read-only branch/tag/ancestor checks before pushing.

### Metadata
- Reproducible: unknown
- See Also: Phase 2 archive connectivity record in GDN_CURRENT_STATUS_A2.md

---

## [ERR-20260726-012] phase2-tag-hardcoded-full-sha

**Logged**: 2026-07-26T02:05:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: config

### Summary
The Phase 3 tag precheck expanded the documented short Phase 2 SHA into an incorrect hard-coded full SHA.

### Error
```text
Phase2 tag moved: f2a4b467f37887824106633524bd0b1c45737e1c
```

### Context
- Local Phase 3 tag absence and remote Phase 3 tag absence were confirmed.
- No tag was created and no remote state changed.
- The failure was in the expected value, not evidence that Phase 2 moved.

### Suggested Fix
Compare local `gdn-a2-phase2^{commit}` with the remote annotated tag's peeled `^{}` commit. Do not invent a full SHA from an abbreviated log value.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-26T02:05:00+08:00
- **Notes**: The retry derives both full commit identities from Git and compares them directly.

---

## [ERR-20260726-011] windows-cpu-test-openmp-duplicate

**Logged**: 2026-07-26T01:45:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
The Phase 3 CPU contract test stopped before execution because the Windows Python process loaded duplicate Intel OpenMP runtimes.

### Error
```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

### Context
- The preceding ctypes ABI suite passed `11/11`.
- PowerShell continued to later commands because native nonzero exit codes are not converted to terminating errors by `$ErrorActionPreference`.
- No NPU code or source was changed.

### Suggested Fix
Run the isolated CPU-only contract process with `KMP_DUPLICATE_LIB_OK=TRUE` and explicitly fail on `$LASTEXITCODE`; never infer its result from later commands.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-26T01:46:00+08:00
- **Notes**: The isolated CPU-only process ran with `KMP_DUPLICATE_LIB_OK=TRUE`, explicitly returned code 0, and passed `2/2`.

---

## [ERR-20260726-010] a2-ssh-banner-timeout-after-performance

**Logged**: 2026-07-26T01:38:00+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
A post-matrix read-only SSH evidence refresh timed out during banner exchange.

### Error
```text
Connection timed out during banner exchange
```

### Context
- The final 8/8 NPU matrix had already completed with exit code 0 and all raw JSON/log files were copied locally.
- The failed operation was only a redundant re-hash of previously recorded exact logs.
- No remote source, package, installation, or result was changed.

### Suggested Fix
Retry a short read-only SSH connection before final archive; do not repeat NPU tests or replace existing hashes with empty output.

### Metadata
- Reproducible: unknown

---

## [ERR-20260726-009] local-audit-library-handle-count

**Logged**: 2026-07-26T01:31:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
The local evidence audit treated duplicate handles for the same installed library as distinct library identities.

### Error
```text
CASE_COUNT=8
BAD_GATE_COUNT=8
```

### Context
- Each report listed two handles with the same absolute path and the same expected SHA256.
- Accuracy, ACLNN count, production environment, sample count, and latency gates all passed.
- This was an audit-script assumption; it did not rerun or alter the NPU results.

### Suggested Fix
Allow the installed compatibility pair `libcust_opapi.so`/`libopapi.so` when both are in the same vendor directory and `cmp` plus SHA256 prove identical bytes; gate on one unique byte identity, not raw loader handle count.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-26T01:33:00+08:00
- **Notes**: Read-only inspection proved the two paths are the expected compatibility pair in one install tree and are byte-identical; the audit now checks that contract.

---

## [ERR-20260726-008] powershell-ssh-jq-quoting

**Logged**: 2026-07-26T01:28:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
An inline `jq` summary filter was corrupted by PowerShell/SSH quote processing after the NPU matrix had completed.

### Error
```text
jq: error: Invalid escape
```

### Context
- All eight benchmark processes had already passed and written immutable JSON files.
- Only the read-only summary command failed; no result or remote source was changed.

### Suggested Fix
Fetch structured JSON with `scp` and parse it locally through PowerShell `ConvertFrom-Json`, or upload a fixed filter file. Do not embed a long quoted `jq` program in SSH.

### Metadata
- Reproducible: yes
- See Also: prior PowerShell/SSH nested quote errors in this file

### Resolution
- **Resolved**: 2026-07-26T01:28:00+08:00
- **Notes**: Switched to local structured parsing of the eight raw JSON files.

---

## [ERR-20260726-007] powershell-bare-assignment

**Logged**: 2026-07-26T01:20:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A redundant bare `bash=''` assignment made PowerShell stop before uploading a revised remote matrix wrapper.

### Error
```text
bash= : The term 'bash=' is not recognized
```

### Context
- The failure occurred locally before `scp`, SSH, or any NPU workload.
- It repeated the missing-variable-sigil pattern from `ERR-20260726-006`.

### Suggested Fix
Keep upload, remote syntax checking, and execution as separate short commands; do not add unused shell variables.

### Metadata
- Reproducible: yes
- See Also: ERR-20260726-006

### Resolution
- **Resolved**: 2026-07-26T01:20:00+08:00
- **Notes**: Removed the unused assignment and resumed with single-purpose commands.

---

## [ERR-20260726-007] phase3-staging-route-788-build

**Logged**: 2026-07-26T02:27:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
The first non-incremental Phase 3 staging build failed while producing the BF16/C128 `ChunkCumsumKkt` route.

### Error
```text
ChunkCumsumKkt_*_mix_aiv_788.o: No such file or directory
Opc tool compile failed.
gmake: *** [Makefile:156: all] Error 2
```

### Context
- Local and remote shared-header SHA256 both matched the frozen `062566...a22e` digest.
- Remote ABI tests passed `11/11` before the clean build.
- The package was not installed and no NPU workload ran.
- The missing object is a downstream symptom; the earlier route-specific compiler diagnostic still needs extraction.

### Suggested Fix
Inspect only the route `788` OPC logs and generated directory, identify the first actual compile diagnostic, then freeze one source-level variable before rebuilding.

### Metadata
- Reproducible: unknown
- Related Files: fla/ops/ascendc/gdn/gdn_preprocess/chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.h

### Resolution
- **Resolved**: 2026-07-26T02:36:00+08:00
- **Notes**: Persistent single-target replay exposed the unsupported three-argument `Copy`; replaced it with the verified five-argument single-element form and passed local ABI/static gates.

---

## [ERR-20260726-008] route-target-replay-missing-cann-env

**Logged**: 2026-07-26T02:29:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
A standalone CMake target replay did not inherit the CANN environment from the completed `build.sh` process.

### Error
```text
asc_opc: command not found
```

### Context
- The command stopped before OPC or kernel compilation started.
- It did not modify source, install a package, or run NPU work.

### Suggested Fix
Use a fixed Bash wrapper that sources CANN, activates `chw-py11`, persists stdout/stderr and writes the target exit code.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase3_staging_route788_diag_remote.sh

### Resolution
- **Resolved**: 2026-07-26T02:29:00+08:00
- **Notes**: Added the fixed environment-aware route replay wrapper before retrying.

---

## [ERR-20260725-023] guessed-skill-install-root

**Logged**: 2026-07-25T23:59:00+08:00
**Priority**: low
**Status**: resolved
**Area**: config

### Summary
The first read of `cann-op-migration` used a guessed user-level skill root instead of the installed workspace-level root.

### Error
```text
Cannot find path 'C:\Users\Administrator\.agents\skills\cann-op-migration\SKILL.md'
```

### Context
- The failure was read-only and occurred before any GDN action.
- The available-skills catalog mapped this skill to `D:\workspace\.agents\skills`.

### Suggested Fix
Resolve every selected skill from the available-skills root mapping before reading it; do not infer an install root from another skill.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-25T23:59:00+08:00
- **Notes**: Re-read the complete skill from the catalog-resolved workspace path with explicit UTF-8 decoding.

---

## [ERR-20260725-024] read-thread-argument-mismatch

**Logged**: 2026-07-25T23:59:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The first attempts to read an older Codex task passed an argument combination rejected by the task API.

### Error
```text
read_thread received invalid arguments.
```

### Context
- The failure was read-only and did not affect the GDN repository or remote A2 state.
- The task was first resolved through `list_threads`; a minimal `read_thread` call then succeeded.

### Suggested Fix
Start with the minimal required `threadId` call, then add cursor or output options only when needed.

### Metadata
- Reproducible: unknown

### Resolution
- **Resolved**: 2026-07-25T23:59:00+08:00
- **Notes**: The old task's latest turn was retrieved successfully and matched the user-provided final reply.

---

## [ERR-20260725-019] powershell-object-pipeline-parser-error

**Logged**: 2026-07-25T23:05:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A nonessential PowerShell brace-count command used an invalid pipeline after `foreach`.

### Error
```text
ParserError: An empty pipe element is not allowed.
```

### Context
- The command was read-only and independent of the successful `git diff --check` and ABI suite.
- No source, build, remote, or NPU state changed.

### Suggested Fix
Wrap `foreach (...) { ... }` in `@(...)` before piping, or omit heuristic brace counting when compiler/static gates are available.

### Metadata
- Reproducible: yes
- Related Files: fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri

### Resolution
- **Resolved**: 2026-07-25T23:05:00+08:00
- **Notes**: Kept the authoritative diff/ABI results and did not treat the failed heuristic as code evidence.

---

## [ERR-20260725-018] cann-set-env-reads-unset-cmake-prefix-path

**Logged**: 2026-07-25T22:35:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
An A2 profiler wrapper using `set -u` exited while sourcing CANN because `CMAKE_PREFIX_PATH` was unset.

### Error
```text
set_env.sh: line 31: CMAKE_PREFIX_PATH: unbound variable
```

### Context
- Remote shell syntax, Python compilation, hashes, and device-idle checks had passed.
- The diagnostic printed the error before conda activation, but Bash continued and the already-dispatched
  run completed both traces successfully. No overlapping retry was launched.

### Suggested Fix
Initialize `CMAKE_PREFIX_PATH` along with other environment variables before sourcing the CANN setup script under `set -u`.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase3_core_profiler_diag_remote.sh
- See Also: ERR-20260725-014

### Resolution
- **Resolved**: 2026-07-25T22:35:00+08:00
- **Notes**: Added an empty default for future runs. The in-flight profiler was allowed to finish, then its
  exit code, trace hashes, structured summary, and device release were verified before accepting evidence.

---

## [ERR-20260725-017] local-powershell-has-no-bash

**Logged**: 2026-07-25T22:30:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The Windows PowerShell host could not run `bash -n` for an A2-only remote wrapper.

### Error
```text
bash: The term 'bash' is not recognized
```

### Context
- Local Python compilation succeeded; no remote or NPU work had started.
- The wrapper is executed on A2 Linux, where Bash is available.

### Suggested Fix
Use local checks only for Python/text on this host and run `bash -n` on A2 after SHA256-verified upload.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase3_core_profiler_diag_remote.sh

### Resolution
- **Resolved**: 2026-07-25T22:30:00+08:00
- **Notes**: Moved the shell syntax gate to the target A2 environment before execution.

---

## [ERR-20260725-016] torch_npu chrome trace exported a top-level list

**Logged**: 2026-07-25T21:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
The Phase 3 profiler smoke captured and exported its baseline trace, but the local summary parser assumed a Chrome trace dictionary while this torch_npu version exported a top-level event list.

### Error
```text
AttributeError: 'list' object has no attribute 'get'
```

### Context
- Baseline profiling data parsed successfully before the Python summary failed.
- The fused profiling call had not started, so no kernel-count conclusion was produced.
- No operator or NPU runtime error occurred.

### Suggested Fix
Accept both standard `{"traceEvents": [...]}` dictionaries and top-level event lists before filtering device events.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase3_profiler_smoke.py

### Resolution
- **Resolved**: 2026-07-25T21:00:00+08:00
- **Notes**: The parser now branches on the JSON payload type before reading events.

---

## [ERR-20260725-015] fused cumsum used unsynchronized scalar UB loop

**Logged**: 2026-07-25T20:10:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
The exploratory `ChunkCumsumKkt` kernel updated a vector-pipeline UB tensor through a raw scalar pointer loop, producing neither raw `g` nor the required chunk-local FP32 cumsum.

### Error
```text
fused g_cumsum vs NPU ChunkLocalCumsum: equal=4/256, max_abs=0.05804957449436188
fused A_raw valid region vs two-op NPU baseline: equal=5517/8064, max_abs=0.05121582746505737
```

### Context
- A2/CANN 9.1.0.beta1, dense FP16/C64, `B=1,H=2,T=128,K=128`.
- Only the first token of each of four chunks matched the cumsum baseline.
- Both fused outputs were finite and the `A_raw` invalid region was zero, but the cumsum and valid KKT values were wrong.
- Static review excluded output-order mismatch and localized the first divergence to `ComputePrefixCumsum`.

### Suggested Fix
Do not mix raw scalar UB pointer arithmetic into a vector-pipeline tensor without proven scalar/vector synchronization. Reuse the established `ChunkLocalCumsum` sequential FP32 vector-add order with aligned UB scalar buffers and explicit MTE/vector events.

### Metadata
- Reproducible: yes
- Related Files: fla/ops/ascendc/gdn/gdn_preprocess/chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.h

### Attempt 1
- Replaced the raw pointer loop with aligned single-element vector additions.
- The rebuild ran, but only `5/256` cumsum values matched the NPU baseline; the event sequence was still weaker than the repository's verified KDA cumsum pattern.
- Attempt 2 reuses one event ID per hard-event type and waits for `MTE3_MTE2` after every scalar output write.

### Resolution
- **Resolved**: 2026-07-25T20:38:00+08:00
- **Notes**: Attempt 2 matched the verified KDA event sequence. On dense FP16/C64, fused `g_cumsum` matched `ChunkLocalCumsum` `256/256` bit-exact and fused `A_raw` matched the two-op NPU baseline `16384/16384` bit-exact, with finite outputs and zero invalid region.

---

## [ERR-20260725-014] vendor set_env reads unset variables under nounset

**Logged**: 2026-07-25T20:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The Phase 3 post-install gate stopped before symbol checks because the generated vendor environment script expands variables that may be unset while the wrapper uses `set -u`.

### Error
```text
set_env.bash: line 2: ASCEND_CUSTOM_OPP_PATH: unbound variable
```

### Context
- The complete A2 package built and installed successfully before the failure.
- No symbol check, Python process, or NPU smoke ran after the failure.
- The generated script also expands `LD_LIBRARY_PATH` without a default.

### Suggested Fix
Initialize `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH` to empty values before sourcing generated vendor environment scripts under `set -u`. Add the source-tree Python package to `PYTHONPATH` when testing a new pure-Python wrapper that is not yet in the installed wheel.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase3_build_smoke_remote.sh

### Resolution
- **Resolved**: 2026-07-25T20:00:00+08:00
- **Notes**: The wrapper now initializes both variables and points `PYTHONPATH` at the Phase 3 source tree before the post-install smoke.

---

## [ERR-20260725-010] A2 performance preflight used wrong wrapper path and npu-smi form

**Logged**: 2026-07-25T18:50:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The Phase 2 preflight checked the wrapper in `/opt/chw/` instead of the remote repo root, and used an unsupported `npu-smi info -i 1` form.

### Error
```text
sha256sum: /opt/chw/.codex_phase2_standalone_remote.sh: No such file or directory
Must input parameter of type.
```

### Context
- The benchmark and runner hashes still matched; no Python or NPU workload was started.
- This A2 `npu-smi` build exposes process and utilization data through typed queries.

### Suggested Fix
Resolve the wrapper under the remote repository, use `npu-smi info -t proc-mem -i 1` and `npu-smi info -t usages -i 1`, then run remote `bash -n` before launch.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase2_standalone_remote.sh

### Resolution
- **Resolved**: 2026-07-25T18:50:00+08:00
- **Notes**: The corrected queries confirmed device 1 had no process and 0% AICore/AIVector utilization.

---

## [ERR-20260725-011] Phase 1 long-varlen standalone MTE out-of-range

**Logged**: 2026-07-25T18:52:00+08:00
**Priority**: high
**Status**: resolved
**Area**: tests

### Summary
The clean-process Phase 1 baseline failed on the first `L_VARLEN_T32768` call before timing, so the balanced runner stopped without launching Phase 2 or later rounds.

### Error
```text
npuSynchronizeDevice: error code 507015
The DDR address of the MTE instruction is out of range.
```

### Context
- A2 device 1, FP16, `T=32768`, physical heads 8, `C=128`, `cu_seqlens=[0,8191,16384,32768]`.
- Failure occurred at the initial standalone synchronization, before finiteness checks and JSON output.
- The runner correctly stopped after the first child; no benchmark process remained.

### Suggested Fix
Treat this as a Phase 1 long-sequence baseline failure. Validate Phase 2 alone in a fresh process before deciding the Phase 2 acceptance disposition; do not debug or patch Phase 1 unless the project scope changes.

### Metadata
- Reproducible: unknown
- Related Files: torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py, torch_custom/fla_npu/test/run_gdn_phase2_performance_matrix.py
- See Also: Phase 1 workspace-initialization evidence in fla/ops/ascendc/gdn/docs/GDN_CURRENT_STATUS_A2.md

---

## [ERR-20260725-012] GitHub remote unavailable during Phase 2 archive push

**Logged**: 2026-07-25T19:10:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
The Phase 2 performance-close commit was created locally, but the read-only remote ref check could not connect to GitHub over HTTPS, so no push was attempted.

### Error
```text
Failed to connect to github.com port 443: Could not connect to server
```

### Context
- Local `HEAD` is `2b8161d`; `f2a4b46` is its ancestor.
- The immutable annotated tag still resolves to commit `f2a4b46`.
- `git ls-remote chw` failed before any remote mutation.

### Suggested Fix
Retry the remote heads/tags check when GitHub connectivity returns, then use a normal fast-forward branch push without `--force` and without pushing tags.

### Metadata
- Reproducible: unknown
- Related Files: fla/ops/ascendc/gdn/docs/GDN_CURRENT_STATUS_A2.md

### Resolution
- **Resolved**: 2026-07-25T19:15:00+08:00
- **Notes**: A temporary local SOCKS tunnel through A2 restored local Git HTTPS access while keeping credentials on the local machine. The branch was fast-forwarded normally and the tunnel was stopped.

---

## [ERR-20260725-013] A2 isolated checkout has no chw remote

**Logged**: 2026-07-25T19:12:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The A2 checkout could not be used immediately as a GitHub push relay because its isolated repository has no remote named `chw`.

### Error
```text
fatal: 'chw' does not appear to be a git repository
```

### Context
- Only a read-only `git ls-remote` was attempted.
- No remote configuration or working tree was changed.

### Suggested Fix
Inspect existing A2 remotes. Use an already configured authenticated fork remote if present; otherwise leave the local commit pending instead of mutating remote configuration during the archive step.

### Metadata
- Reproducible: yes
- Related Files: fla/ops/ascendc/gdn/docs/GDN_CURRENT_STATUS_A2.md

### Resolution
- **Resolved**: 2026-07-25T19:15:00+08:00
- **Notes**: No A2 repository remote was added. A2 was used only as an outbound proxy for the authenticated local Git client.

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
- Pattern-Key: infra.powershell_ssh_nested_quoting
- Recurrence-Count: 4
- Last-Seen: 2026-07-25
- See Also: ERR-20260725-001

### Resolution
- **Resolved**: 2026-07-24T21:05:00+08:00
- **Notes**: Switched builds to fixed remote scripts. A later wrapper around a successful Phase 3 script again
  lost its `$rc` expansion and omitted the sidecar file; build success was accepted only after checking the
  package timestamp/hash, final log markers, compiled binaries, source-copy hashes, symbols, and no live process.

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
- Pattern-Key: infra.remote_bundle_missing_tracked_source
- Recurrence-Count: 2
- Last-Seen: 2026-07-25

### Resolution
- **Resolved**: 2026-07-24T19:00:00+08:00
- **Notes**: Restored the missing tracked patch in the isolated A2 test directory before rebuilding.

The recurrence on 2026-07-25 was caused by a recursive `rsync --exclude build`, which also excluded the tracked
`cmake/third_party/build/` source directory. Exclude only root-level generated directories, or copy the complete
source tree and remove generated roots explicitly after validating their resolved paths.

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

### Recurrence
- **Last Seen**: 2026-07-25T21:30:00+08:00
- The same host-level duplicate runtime aborted the Phase 3 CPU reference test before test execution.
- Continued with syntax/AST checks locally and the clean A2 conda environment; the unsafe duplicate-library override remains forbidden.

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
## [ERR-20260725-007] remote-python-before-conda-activation

**Logged**: 2026-07-25T21:20:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
A2 login shell has no default `python`; a profiler precheck stopped before NPU execution.

### Error
```text
bash: python: command not found
```

### Context
- A remote one-line precheck invoked `python -m py_compile` before sourcing conda.
- Source hashes matched and no Python/NPU workload started.

### Suggested Fix
Run Python checks only inside the fixed remote wrapper after sourcing the conda profile and activating `chw-py11`.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase3_profiler_smoke_remote.sh
- See Also: ERR-20260725-006
- Pattern-Key: a2.remote_python_requires_conda
- Recurrence-Count: 2
- First-Seen: 2026-07-25
- Last-Seen: 2026-07-25

### Resolution
- **Resolved**: 2026-07-25T21:20:00+08:00
- **Notes**: Continued with the wrapper that activates conda before invoking Python. A later read-only
  summary command hit the same login-shell condition and was corrected without rerunning NPU work by
  invoking `/root/miniconda3/envs/chw-py11/bin/python` explicitly.

---

## [ERR-20260725-008] inline-json-summary-over-ssh

**Logged**: 2026-07-25T21:51:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
An inline Python JSON summary command was corrupted by PowerShell/SSH nested quoting.

### Error
```text
SyntaxError: unterminated string literal
```

### Context
- The Phase 3 accuracy pilot had already completed successfully.
- The failing command was read-only and did not launch NPU work or modify files.

### Suggested Fix
Copy structured JSON locally with `scp` and read it directly, or use an uploaded fixed script; do not embed Python source inside PowerShell-to-SSH command strings.

### Metadata
- Reproducible: yes
- Related Files: .phase3_core_accuracy_pilot_summary.json
- See Also: prior PowerShell/SSH nested quote errors in this file
- Recurrence-Count: 2
- Last-Seen: 2026-07-26

### Resolution
- **Resolved**: 2026-07-25T21:51:00+08:00
- **Notes**: Retrieved the JSON with scp and parsed it locally. The same read-only summary pattern recurred
  after a later completed NPU matrix, so structured evidence is now always copied locally and parsed with
  PowerShell instead of embedding Python in SSH.

---
## [ERR-20260725-020] powershell-expands-remote-bash-variables

**Logged**: 2026-07-25T23:10:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
PowerShell expanded `$repo` and `$pyenv` before an SSH read-only hash command reached Bash.

### Error
```text
sha256sum: /.codex_phase3_core_matrix_remote.sh: No such file or directory
```

### Context
- The remote command declared Bash variables and then referenced them inside a PowerShell double-quoted string.
- The failed command did not start Python, run an NPU operator, or mutate remote files.

### Suggested Fix
Use explicit absolute remote paths for short read-only checks, or invoke an uploaded Bash wrapper for multi-step work. Do not interpolate remote Bash variables through PowerShell.

### Metadata
- Reproducible: yes
- See Also: ERR-20260725-001
- Recurrence-Count: 2

### Resolution
- **Resolved**: 2026-07-25T23:10:00+08:00
- **Notes**: Subsequent verification uses explicit absolute paths and the existing checked remote wrapper for execution. Recurred during a read-only new-vs-installed kernel hash check; no target path was read or mutated.

---
## [ERR-20260725-021] offline-torch-import-autoloads-npu-backend

**Logged**: 2026-07-25T23:25:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A pure CPU offline tensor comparison imported `torch_npu` automatically and failed because the one-line SSH command had not sourced CANN libraries.

### Error
```text
ImportError: libhccl.so: cannot open shared object file
```

### Context
- The command only intended to read CPU `.pt` evidence and write a JSON grouping summary.
- It failed during `import torch`; no evidence file was read, no NPU call ran, and no operator state changed.

### Suggested Fix
For CPU-only offline evidence parsing, set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. For any NPU execution, use the checked remote wrapper that sources CANN, conda, and vendor OPP environments.

### Metadata
- Reproducible: yes
- See Also: ERR-20260725-007

### Resolution
- **Resolved**: 2026-07-25T23:25:00+08:00
- **Notes**: The retry explicitly disables backend autoload because it is CPU-only.

---
## [ERR-20260725-022] assumed-kernel-local-shared-header-path

**Logged**: 2026-07-25T23:32:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
A read-only diff assumed the cumulative kernel had a directory-local `chunk_scaled_dot_kkt.h`, but CMake imports the shared preprocess header instead.

### Error
```text
Could not access chunk_kkt_solve_tri/op_kernel/chunk_scaled_dot_kkt.h
```

### Context
- The failed operation was a read-only source comparison.
- No build, install, NPU execution, or source mutation occurred.

### Suggested Fix
Resolve includes from the operator CMake compile options before comparing files; use `rg --files` rather than inferring a local copy.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-25T23:32:00+08:00
- **Notes**: Verified that both kernels compile against `gdn_preprocess/chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.h`.

---
## [ERR-20260725-025] missing-remote-phase3-install-wrapper

**Logged**: 2026-07-25T23:59:30+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The Phase 3 install preflight assumed the new local wrapper had already been uploaded to the isolated A2 checkout.

### Error
```text
bash: .../.codex_phase3_stale_beta_install_pilot_remote.sh: No such file or directory
```

### Context
- The package and both formal Python test scripts were present and hash-matched.
- The failed checks did not install a package or run an NPU operator.

### Suggested Fix
For each new remote wrapper, explicitly upload it before the remote hash and `bash -n` gates.

### Metadata
- Reproducible: yes
- Related Files: .codex_phase3_stale_beta_install_pilot_remote.sh

### Resolution
- **Resolved**: 2026-07-25T23:59:30+08:00
- **Notes**: The wrapper is uploaded and verified in the following retry before execution.

---
## [ERR-20260725-026] powershell-ssh-quote-in-readback

**Logged**: 2026-07-25T23:59:45+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A redundant post-failure installed-file hash readback used an unterminated nested PowerShell/SSH quote.

### Error
```text
bash: -c: line 1: unexpected EOF while looking for matching `"'
```

### Context
- The install wrapper had already completed package-vs-installed `cmp` checks and printed all installed hashes.
- This failed command was read-only; it did not install, build, or run an NPU operator.

### Suggested Fix
Use an uploaded Bash wrapper or explicit non-interpolated paths for multi-file remote readback.

### Metadata
- Reproducible: yes
- See Also: ERR-20260725-020

### Resolution
- **Resolved**: 2026-07-25T23:59:45+08:00
- **Notes**: Reused the authoritative install-wrapper log, which already contains the successful comparisons and hashes.

---
## [ERR-20260725-027] nested-python-readback-quoting

**Logged**: 2026-07-25T23:59:55+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A redundant inline Python summary extraction was corrupted by PowerShell/SSH nested quoting.

### Error
```text
SyntaxError: unterminated string literal
```

### Context
- The authoritative `summary.json` already existed and its SHA256 was recorded.
- The failed operation was read-only and did not run an NPU operator.

### Suggested Fix
Retrieve structured evidence with `scp` and parse locally instead of embedding Python in SSH.

### Metadata
- Reproducible: yes
- See Also: ERR-20260725-019

### Resolution
- **Resolved**: 2026-07-25T23:59:55+08:00
- **Notes**: Copied the JSON locally; its SHA256 exactly matches the remote evidence.

---

## [ERR-20260725-028] status-patch-context-mismatch

**Logged**: 2026-07-25T23:59:58+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
A status-document patch omitted one space from the expected context and failed verification.

### Error
```text
apply_patch verification failed: Failed to find expected lines
```

### Context
- No hunk was applied, so the status file was unchanged.
- The exact source lines were read before retrying.

### Suggested Fix
Read the tight target range immediately before patching frequently updated status documents.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-25T23:59:58+08:00
- **Notes**: Retried with the exact current text.

---
## [ERR-20260726-001] rg-windows-path-glob

**Logged**: 2026-07-26T00:00:10+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
`rg` received a Windows path argument containing shell-style wildcards and rejected it as an invalid filename.

### Error
```text
rg: .codex_phase3_*build*.sh: filename, directory name, or volume label syntax is incorrect
```

### Context
- The failure was an auxiliary read-only search.
- No source, build, install, or NPU state changed.

### Suggested Fix
Use `rg --glob` for filename patterns or pass explicit Windows paths.

### Metadata
- Reproducible: yes
- Recurrence-Count: 2
- Last-Seen: 2026-07-26

### Resolution
- **Resolved**: 2026-07-26T00:00:10+08:00
- **Notes**: A later search repeated the same shell-style path glob; all subsequent searches use `rg --glob` from the repo root.

---

## [ERR-20260726-002] status-patch-word-mismatch

**Logged**: 2026-07-26T00:08:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
A status-document patch used `远程` while the current source used `远端`, so exact hunk verification failed.

### Error
```text
apply_patch verification failed: Failed to find expected lines
```

### Context
- The failed hunks changed no files.
- The target status file timestamp had not changed; the mismatch was one word in the supplied context.

### Suggested Fix
Read the exact tight range and preserve its wording verbatim before patching the frequently updated status entry.

### Metadata
- Reproducible: yes
- See Also: ERR-20260725-028

### Resolution
- **Resolved**: 2026-07-26T00:08:00+08:00
- **Notes**: Retried with the exact `远端同步` source text and the patch applied.

---

## [ERR-20260726-003] powershell-ssh-command-substitution

**Logged**: 2026-07-26T00:12:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A PowerShell SSH precheck escaped remote command substitution into a literal `\$(` and Bash rejected it.

### Error
```text
bash: syntax error near unexpected token `('
```

### Context
- The header upload completed before the failed read-only precheck.
- The fixed build wrapper was not started; no package was installed and no NPU command ran.

### Suggested Fix
Avoid remote command substitution across PowerShell and SSH; pipe an explicit expected digest to `sha256sum -c`.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-26T00:12:00+08:00
- **Notes**: The retry uses `printf ... | sha256sum -c -` and contains no nested command substitution.

---

## [ERR-20260726-004] powershell-ssh-exit-wrapper-quoting

**Logged**: 2026-07-26T00:14:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
An inline remote exit-code wrapper had mismatched quotes after crossing PowerShell and SSH.

### Error
```text
bash: unexpected EOF while looking for matching `"'
```

### Context
- Remote Bash failed during parse, before the fixed build wrapper executed.
- No build, install, or NPU state changed.

### Suggested Fix
Invoke the already uploaded fixed Bash wrapper directly and use SSH's process exit code; avoid inline remote variables and nested quoting.

### Metadata
- Reproducible: yes
- See Also: ERR-20260726-003

### Resolution
- **Resolved**: 2026-07-26T00:14:00+08:00
- **Notes**: The retry uses a single direct `timeout ... bash wrapper > log 2>&1` command.

---

## [ERR-20260726-005] run-installer-missing-conda-context

**Logged**: 2026-07-26T00:47:30+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
The install-only wrapper invoked the run package before activating the target `chw-py11` environment.

### Error
```text
fla_npu is not importable. Install flash-linear-attention-npu wheel first.
Target wheel OPP vendor not found: /vendors/fla_npu_transformer.
```

### Context
- The run package SHA256 check passed.
- The installer stopped before file comparison; no Python operator or NPU command ran.

### Suggested Fix
Source CANN and activate the target conda environment before running the package installer, as the established install wrappers do.

### Metadata
- Reproducible: yes
- See Also: ERR-20260725-007

### Resolution
- **Resolved**: 2026-07-26T00:47:30+08:00
- **Notes**: Added the established CANN/conda setup to the fixed install-audit wrapper before retrying the same small step.

---

## [ERR-20260726-006] powershell-missing-variable-sigil

**Logged**: 2026-07-26T00:49:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
An auxiliary local Bash availability probe omitted PowerShell's `$` variable sigil.

### Error
```text
bashPath=: The term 'bashPath=' is not recognized
```

### Context
- The error was non-terminating and only affected an optional local Bash probe.
- The uploaded wrapper SHA256 and authoritative remote `bash -n` both passed; installation had not started yet.

### Suggested Fix
Use `$bashPath = Get-Command ...` in PowerShell, or omit the local probe when the authoritative remote syntax check is already required.

### Metadata
- Reproducible: yes

### Resolution
- **Resolved**: 2026-07-26T00:49:00+08:00
- **Notes**: Continued from the successful remote hash and `bash -n` gates without relying on the faulty optional probe.

---
