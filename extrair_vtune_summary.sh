#!/bin/bash
# extrair_vtune_summary_fixed.sh
BASE_DIR="$HOME/vtune_results"
OUTPUT_CSV="$BASE_DIR/summary_table_spread_fixed.csv"

echo "Threads,Samples,Repetition,CPI Rate,Effective Physical Core Utilization (%),Memory Bound (%),Cache Bound (%)" > "$OUTPUT_CSV"

for dir in "$BASE_DIR"/vtune_spread_5_t*_s*_r*; do
    summary_file="$dir/summary.txt"
    if [ -f "$summary_file" ]; then
        # extrai threads, samples, rep a partir do nome do diretório
        threads=$(echo "$dir" | sed -E 's/.*_t([0-9]+)_s([0-9]+)_r([0-9]+)/\1/')
        samples=$(echo "$dir" | sed -E 's/.*_t([0-9]+)_s([0-9]+)_r([0-9]+)/\2/')
        rep=$(echo "$dir" | sed -E 's/.*_t([0-9]+)_s([0-9]+)_r([0-9]+)/\3/')

        # Função auxiliar para extrair primeiro número antes de '%' ou primeiro número entre parênteses
        extract_percent_or_paren() {
            local file="$1"
            local pattern="$2"
            # tenta número seguido de '%' (ex: 15.1%)
            val=$(grep -m1 "$pattern" "$file" | awk '{
                if (match($0, /([0-9]+(\.[0-9]+)?)%/, a)) {
                    print a[1]; exit
                }
                # se não encontrou %, tenta número entre parênteses
                if (match($0, /\(([0-9]+(\.[0-9]+)?)/, b)) {
                    print b[1]; exit
                }
                # se não encontrar, tenta primeiro número simples na linha
                if (match($0, /([0-9]+(\.[0-9]+)?)/, c)) {
                    print c[1]; exit
                }
            }')
            # se vazio, devolve NA
            if [ -z "$val" ]; then
                echo "NA"
            else
                echo "$val"
            fi
        }

        # CPI Rate -> número simples (sem %)
        cpi=$(awk -F':' '/CPI Rate/ { if (match($0, /([0-9]+(\.[0-9]+)?)/, a)) print a[1]; exit }' "$summary_file")
        cpi=${cpi:-NA}

        # Effective Physical Core Utilization -> prefer percent (15.1%), senão número entre parênteses
        core_util=$(extract_percent_or_paren "$summary_file" "Effective Physical Core Utilization")
        mem_bound=$(extract_percent_or_paren "$summary_file" "Memory Bound")
        cache_bound=$(extract_percent_or_paren "$summary_file" "Cache Bound")

        echo "$threads,$samples,$rep,$cpi,$core_util,$mem_bound,$cache_bound" >> "$OUTPUT_CSV"
    fi
done

echo "✅ Extração concluída: $OUTPUT_CSV"
