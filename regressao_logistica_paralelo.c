#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#define N 900
#define P 12
#define Nnew 70
void iniciar_parametros(double B[P + 1], double valor_inicial) {
    for (int i = 0;i < P + 1;i++)
        B[i] = valor_inicial;
}
void iniciar_dados_de_treino(double X[N][P], double Y[N]) {

    FILE* fp = fopen("Cardiovascular_Disease_Dataset.csv", "r");
    if (fp == NULL) {
        perror("Erro ao abrir o arquivo");
        // return 1;
    }
    char linha[1024];
    int i = 0;

    // ler a primeira linha (cabeçalho) e descartar
    fgets(linha, sizeof(linha), fp);

    // ler até 1000 linhas ou até acabar arquivo
    while (i < N && fgets(linha, sizeof(linha), fp)) {
        char* token;
        int j = 0;

        // pegar primeiro campo (id) e ignorar
        token = strtok(linha, ",");
        token = strtok(NULL, ","); // vai para o campo idade

        // processar atributos (idade até penúltimo campo)
        while (token != NULL && j <= P) {
            if (j == P) {
                // último atributo da linha é o target
                Y[i] = atof(token);
            }
            else {
                X[i][j] = atof(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }

    fclose(fp);






}
void combinacao_linear(double X[N][P], double B[P + 1], double Z[N]) {
    #pragma omp for schedule(static)
    for (int i = 0; i < N; ++i) {
        double s = B[0];
        for (int j = 0; j < P; ++j) s += B[j+1] * X[i][j];
        Z[i] = s;
    }
}
void sigmoide(double p[N], double Z[N]) {
    #pragma omp for schedule(static)
    for (int i = 0; i < N; ++i)
        p[i] = 1.0f / (1.0f + expf(-Z[i]));
}
void gradiente(double p[N], double Y[N], double X[N][P],double *loc) {
    for (int j = 0; j < P+1; ++j) loc[j] = 0.0f;

    #pragma omp for schedule(static)
    for (int i = 0; i < N; ++i) {
        double err = p[i] - Y[i];
        loc[0] += err;
        for (int j = 0; j < P; ++j)
            loc[j+1] += err * X[i][j];
    }
}

void atualizar_parametros(double B[P + 1], double gradJ_B[P + 1], double alfa) {
    for (int j = 0;j < P + 1;j++) {
        B[j] -= alfa * gradJ_B[j];
    }
}
void imprimir_parametros(double B[P + 1]) {
    for (int j = 0;j < P + 1;j++) {
        printf("Parametro %d: %.2f\n", j, B[j]);
    }
}
void iniciar_dados_de_teste(double X[Nnew][P], double Y[Nnew]) {

    FILE* fp = fopen("Cardiovascular_Disease_Dataset.csv", "r");
    if (fp == NULL) {
        perror("Erro ao abrir o arquivo");
        // return 1;
    }
    char linha[1024];
    int i = 0;

    // ler a primeira linha (cabeçalho) e descartar
    for (int i = 0;i < N + 10;i++)
        fgets(linha, sizeof(linha), fp);

    // ler até 1000 linhas ou até acabar arquivo
    while (i < Nnew && fgets(linha, sizeof(linha), fp)) {
        char* token;
        int j = 0;

        // pegar primeiro campo (id) e ignorar
        token = strtok(linha, ",");
        token = strtok(NULL, ","); // vai para o campo idade

        // processar atributos (idade até penúltimo campo)
        while (token != NULL && j <= P) {
            if (j == P) {
                // último atributo da linha é o target
                Y[i] = atof(token);
            }
            else {
                X[i][j] = atof(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }

    fclose(fp);
}
void imprimir_resultados(double X[Nnew][P], double B[P + 1], double Y[Nnew]) {
    printf("Resultados de teste:\n");
    int acertos = 0;
    for (int i = 0;i < Nnew;i++) {
        double Zcalculado = B[0];
        for (int j = 0;j < P;j++) {
            Zcalculado += B[j + 1] * X[i][j];
        }
        double Ycalculado = 1.0 / (1.0 + expf(-Zcalculado));
        printf("Ycalculado: %.2f Yreal: %.2f\n", Ycalculado, Y[i]);
        if (fabs(Ycalculado - Y[i]) < 0.1)
            acertos++;

    }
    printf("Acuracia: %.2f\n", (double)acertos / Nnew);

}

int main() {
    double X[N][P];
    double Y[N];
    double Z[N];
    double p[N];
    double B[P + 1];
    double gradJ_B[P + 1] = { 0 };
    iniciar_parametros(B, 0.1);//
    iniciar_dados_de_treino(X, Y);//


    struct timeval start, end;
    gettimeofday(&start, NULL);
    
/* 
    // define padding stride (maior que P+1 para evitar false sharing)
const int stride = 16; // >= P+1 (13), 16 doubles = 128 bytes, bom alinhamento

#pragma omp parallel default(none) shared(X,Y,Z,p,B,gradJ_B,stride) 
{
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    static double* local_all;
    #pragma omp single
    {
        // aloca com padding
        local_all = malloc((size_t)nthreads * stride * sizeof(double));
        if (!local_all) { printf("malloc falhou\n"); exit(1); }
        // inicializa (opcional)
        for (int t = 0; t < nthreads; ++t)
            for (int j = 0; j < stride; ++j)
                local_all[t*stride + j] = 0.0;
    }
    #pragma omp barrier
    double *loc = local_all + (size_t)tid * stride;

    for (int iter = 0; iter < 500000; ++iter) {

        // zera buffer local (por thread)
        for (int j = 0; j < P+1; ++j) loc[j] = 0.0;

        // UMA SÓ PASSAGEM: combinacao_linear + sigmoide + acumula grad local
        #pragma omp for schedule(static)
        for (int ii = 0; ii < N; ++ii) {
            double s = B[0];
            for (int j = 0; j < P; ++j) s += B[j+1] * X[ii][j];
            double pi = 1.0 / (1.0 + exp(-s));
            p[ii] = pi;
            double err = pi - Y[ii];
            loc[0] += err;
            for (int j = 0; j < P; ++j) loc[j+1] += err * X[ii][j];
        } // fim for ii

        // single reduction e atualiza B
        #pragma omp single
        {
            // soma buffers
            for (int j = 0; j < P+1; ++j) gradJ_B[j] = 0.0;
            for (int t = 0; t < nthreads; ++t) {
                double* lt = local_all + (size_t)t * stride;
                for (int j = 0; j < P+1; ++j) gradJ_B[j] += lt[j];
            }
            // média e update
            for (int j = 0; j < P+1; ++j) {
                gradJ_B[j] /= (double)N;
                B[j] -= 0.01 * gradJ_B[j];
            }
        } // single -> implicit barrier after single
    } // iter
    #pragma omp single
    free(local_all);
}//*/

    ///*
    #pragma omp parallel default(none) shared(X,Y,Z,p,B,gradJ_B)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        static double* local_all;
        #pragma omp single
        local_all = malloc((size_t)nthreads * (P + 1) * sizeof(double));

        double* loc = local_all + tid * (P + 1);

        for (int i = 0; i < 500000; ++i) {
            combinacao_linear(X, B, Z);
            sigmoide(p, Z);
            gradiente(p, Y, X, loc);

            #pragma omp single
            {
                for (int j = 0; j < P + 1; ++j) gradJ_B[j] = 0.0f;
                for (int t = 0; t < nthreads; ++t) {
                    double* lt = local_all + t * (P + 1);
                    for (int j = 0; j < P + 1; ++j)
                        gradJ_B[j] += lt[j];
                }
                for (int j = 0; j < P + 1; ++j) gradJ_B[j] /= (double)N;
                double alfa = 0.01f;
                atualizar_parametros(B, gradJ_B, alfa);
            }
        }
        #pragma omp single
        free(local_all);
    }//*/
    gettimeofday(&end, NULL);
    imprimir_parametros(B);//
    double Xteste[Nnew][P];
    double Yteste[Nnew];
    iniciar_dados_de_teste(Xteste, Yteste);//
    imprimir_resultados(Xteste, B, Yteste);//
    long segundos = end.tv_sec - start.tv_sec;
    long micros = end.tv_usec - start.tv_usec;
    double tempo_total = segundos + micros / 1e6;
    printf("Tempo: %.6f segundos\n", tempo_total);
    return 0;
}