# Trabalho de OpenMP de Programação Distríbuida e Paralela (Regressão Logística)

## Como executar na máquina hype do PCAD

Use o comando sbatch Final.slurm, observe os caminhos dos arquivos necessários pelo arquivo .slurm e adapte se necessário. 
Ele gera resultado para apenas um modo de OMP_PROC_BIND, mude o modo para o desejado e mude o nome dos arquivos gerados para não sobreescrever.
Execute extrair_vtune_summary.sh após rodar o script (também mude de spread para close no arquivo .sh) para gerar um csv com os dados do summary do VTUNE

Para compilar: gcc regressao_logistica_parallel.c -o <NOME_EXECUTAVEL> -fopenmp -lm

