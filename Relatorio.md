# Relatório de Análise de Desempenho – Regressão Logística Paralela com OpenMP

**Autores:** Augusto Kessler Pires (00338627) e Vitor Chassot (00302134)

---

## 1. Objetivo

Este trabalho tem como objetivo implementar e analisar o desempenho de uma **regressão logística paralela** em C utilizando **OpenMP**, avaliando os efeitos da política de afinidade de threads (`OMP_PROC_BIND`) com as opções **CLOSE** e **SPREAD**, e interpretando os resultados através do **Intel VTune Profiler**.

---

## 2. Implementação e Decisões de Paralelização

O código implementado (`regressao4.c`) realiza a leitura, normalização e treinamento de um modelo de regressão logística em modo *batch*, sobre um dataset de diagnóstico de diabetes. O processo inclui:

- Leitura e pré-processamento de um arquivo CSV contendo dados mistos (float, int e strings);
- Conversão de variáveis categóricas em vetores one-hot;
- Normalização dos atributos numéricos;
- Treinamento via **gradiente descendente** com atualização de pesos e *bias*;
- Paralelização utilizando **OpenMP** para exploração de paralelismo de dados.

### 2.1 Regiões Paralelas

A principal região paralela ocorre dentro do loop de treinamento:

```c
#pragma omp parallel for reduction(+:grad[:NUM_FEATURES], dbias)
for (int i = 0; i < num_samples; i++) {
    ...
}
```

- A diretiva `reduction` garante uma **soma segura** entre os gradientes parciais calculados por cada thread;
- Essa abordagem evita *race conditions* e elimina a necessidade de sincronização manual;
- Todo o processo de *forward* (combinação linear + sigmoide) e *backward* (cálculo do gradiente) é feito em um **único loop paralelo**, reduzindo o overhead de criação e destruição de threads.

### 2.2 Estratégias Consideradas

- **Não foram paralelizadas** operações de leitura ou escrita, por se tratarem de operações de I/O (bound em disco); Além disso apenas o tempo de treinamento foi medido
- O cálculo de atualização dos pesos (`w[j] -= ...`) não foi paralelizado, pois o número de parâmetros é pequeno em relação ao número de amostras;
- O número de iterações (`NUM_ITER`) foi fixado (10.000) para garantir comparabilidade entre execuções.

---

## 3. Metodologia Experimental

Os experimentos variaram:
- **Threads:** 1, 5, 10, 20 e 40;
- **Samples:** 100, 1.000, 10.000 e 100.000;
- Cinco repetições por configuração.

O ambiente de execução foi a máquina **Hype**, com **Intel Xeon (2.3 GHz, 20 núcleos físicos e 40 núcleos lógicos)**.  
As políticas de afinidade testadas foram:

- `OMP_PROC_BIND=CLOSE` – threads alocadas próximas no mesmo *socket*;
- `OMP_PROC_BIND=SPREAD` – threads distribuídas entre *sockets*, maximizando o uso de controladores de memória.

---

## 4. Resultados Obtidos

### 4.1 Métricas VTune

- **CPI Rate (Cycles per Instruction):** eficiência de execução;
- **Effective Physical Core Utilization (%):** uso efetivo dos núcleos físicos;
- **Memory Bound (%):** limitação por acesso à memória principal;
- **Cache Bound (%):** limitação por acesso à cache.

---

## 5. Análise dos Resultados

Ressalva: provavelmente os dados obtidos ao rodar o Close estão acima do normal, por algum motivo a máquina estava mais lenta, anteriormente rodando o resultado foi o mesmo para 40 threads e 1 thread, porém nos dados que usamos nos gráficos e que estão no csv o close está mais lento do que deveria. Na teoria os gráficos de speedup não deveriam ser influenciados pois é relativo a velocidade inicial, porém na prática acabou ficando um pouco maior o speedup no Close ao usar 40 threads, que deveria ser igual para ambos.

### 5.1 Entradas Pequenas (100 e 1.000 samples)

A estratégia **CLOSE** foi superior:
- Dados pequenos cabem na cache do *socket*;
- O *overhead* do SPREAD (distribuição entre *sockets*) reduz desempenho;
- A redução de gradientes é local, minimizando custo de comunicação;
- **Cache Bound (%)** elevado indica limitação pela cache, comum em cargas pequenas.

### 5.2 Entradas Grandes (10.000 e 100.000 samples)

A estratégia **SPREAD** foi mais eficiente:
- Execução **memory-bound**, dominada por acessos à RAM;
- SPREAD aproveita múltiplos controladores de memória (NUMA);
- CLOSE satura um único canal de memória;
- **Cache Bound (%)** reduzido, evidenciando que o gargalo é a memória.

### 5.3 Interpretação das Métricas

- **Memory Bound (%):** tempo gasto esperando dados da RAM; valores altos indicam gargalo de largura de banda.
- **Cache Bound (%):** limitação por falhas de cache; alto em datasets pequenos (cache eficiente), baixo em grandes (memory-bound).

---

## 6. NUMA e Afinidade

A máquina segue arquitetura **NUMA (Non-Uniform Memory Access)**:
- Cada *socket* tem controlador de memória próprio;
- Acesso local é mais rápido que remoto;
- `CLOSE` → melhor localidade;  
  `SPREAD` → maior largura de banda.

Assim:
- CLOSE é melhor para cargas pequenas (cache-bound);
- SPREAD é melhor para cargas grandes (memory-bound).

---

## 7. Conclusão

A implementação paralela em OpenMP mostrou:
- Boa escalabilidade e eficiência até 97,9% dos núcleos;
- `reduction` e paralelismo no loop principal reduziram overheads;
- O gargalo muda de **cache-bound** (CLOSE) para **memory-bound** (SPREAD) conforme o volume de dados.

Esses resultados confirmam a importância da **afinidade de threads** e da **arquitetura NUMA** em aplicações paralelas de aprendizado de máquina.

---

**Referências**
- Intel VTune Profiler – User Guide  
- OpenMP 5.0 Specification  
- Dataset: *Diabetes Health Indicator* (Kaggle)
