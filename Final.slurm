#!/bin/bash
#SBATCH --job-name=logistica_vtune_csv
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:40:00
#SBATCH --output=logistica_vtune_%j.out
#SBATCH --error=logistica_vtune_%j.err

source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh
export OMP_PROC_BIND=CLOSE

PROGRAM="$HOME/PPD/Enviar/regressao4"
SRC="${PROGRAM}.c"
CSV_FILE="resultados_vtune_close_5.csv"

# Compilação
gcc -fopenmp -O3 -o $PROGRAM $SRC -lm

# Cabeçalho do CSV
echo "Threads,Samples,Repetition,Time,Loss" > $CSV_FILE

THREADS_LIST=(1 5 10 20 40)
NUM_ROWS=(100 1000 10000 100000)
REPEAT=5

for t in "${THREADS_LIST[@]}"; do
  for s in "${NUM_ROWS[@]}"; do
    for r in $(seq 1 $REPEAT); do
      RESULT_DIR="$HOME/vtune_results/vtune_close_5_t${t}_s${s}_r${r}"
      mkdir -p "$RESULT_DIR"

      echo "Rodando programa NORMAL para CSV: threads=$t, samples=$s, rep=$r"
      $PROGRAM $s $t $r >> "$CSV_FILE"

      echo "Rodando VTune para análise: threads=$t, samples=$s, rep=$r"
      VTUNE_LOG="$RESULT_DIR/vtune_close_5_run.log"
      vtune -collect hpc-performance -result-dir "$RESULT_DIR" $PROGRAM $s $t $r > /dev/null 2> "$VTUNE_LOG"

      # Gerar relatórios
      vtune -report summary  -result-dir "$RESULT_DIR" -format text -report-output "$RESULT_DIR/summary.txt"
      vtune -report hotspots -result-dir "$RESULT_DIR" -format text -report-output "$RESULT_DIR/hotspots.txt"
    done
  done
done