#!/usr/bin/env bash
# Count ARM DSP (SIMD) instructions inside the CMSIS-NN code of a built ELF.
#
# Task C diagnostic. CMSIS-NN kernels compile whether or not ARM_MATH_DSP is
# defined -- without it they silently take a plain-C fallback path, which is the
# leading explanation for "CMSIS-NN symbols are linked but there is no speedup".
# Counting the actual SIMD instructions is what distinguishes the two cases.
#
# Usage: harness/dsp_symbols.sh <path-to.elf>
set -euo pipefail

ELF="${1:?usage: dsp_symbols.sh <elf>}"
OD=$(find "$HOME/Library/Arduino15/packages/arduino/tools" \
        -name "arm-none-eabi-objdump" 2>/dev/null | head -1)
[ -n "$OD" ] || { echo "arm-none-eabi-objdump not found" >&2; exit 1; }

DSP='\b(smlad|smladx|smlald|smuad|smusd|sxtb16|uxtb16|pkhbt|pkhtb|smlabb|smlatt|qadd|qsub|sadd8|ssub8)\b'

DIS=$(mktemp)
"$OD" -d "$ELF" > "$DIS"

total=$(grep -icE "$DSP" "$DIS" || true)
echo "ELF            : $ELF"
echo "total DSP instr: $total"

# Per-function counts across every CMSIS-NN symbol, not just the top-level
# kernels -- arm_convolve_s8 delegates its inner loop to helpers such as
# arm_nn_mat_mult_kernel_s8_s16, so counting only the entry point understates it.
# NOTE: awk uses POSIX ERE, which has no \b word boundary -- an earlier version
# passed grep's pattern straight through and silently matched nothing. Compare
# whole tokens against a mnemonic set instead.
awk '
  BEGIN {
    split("smlad smladx smlald smuad smusd sxtb16 uxtb16 pkhbt pkhtb " \
          "smlabb smlatt qadd qsub sadd8 ssub8", m, " ")
    for (i in m) want[m[i]] = 1
  }
  /^[0-9a-f]+ <.*>:$/ { fn = $2; gsub(/[<>:]/, "", fn); cur = fn; next }
  /^$/ { cur = "" }
  cur != "" {
    for (i = 1; i <= NF; i++) {
      tok = tolower($i)
      sub(/\..*$/, "", tok)          # strip conditional/width suffixes
      if (tok in want) { c[cur]++; break }
    }
  }
  END { for (f in c) if (c[f] > 0) printf "  %-44s %d\n", f, c[f] }
' "$DIS" | sort -k2 -nr | head -14

nm -C "$ELF" 2>/dev/null | grep -coE "arm_[a-z0-9_]*_s8" | xargs echo "cmsis-nn s8 symbols:"
rm -f "$DIS"
