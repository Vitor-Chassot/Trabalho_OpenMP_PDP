#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define N 900
#define P 12
#define Nnew 70
void iniciar_parametros(double B[P+1], double valor_inicial){
  for(int i=0;i<P+1;i++)
    B[i]=valor_inicial;
}
void iniciar_dados_de_treino(double X[N][P], double Y[N]){
    
    FILE *fp = fopen("Cardiovascular_Disease_Dataset.csv", "r");
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
        char *token;
        int j = 0;

        // pegar primeiro campo (id) e ignorar
        token = strtok(linha, ",");
        token = strtok(NULL, ","); // vai para o campo idade

        // processar atributos (idade até penúltimo campo)
        while (token != NULL && j <= P) {
            if (j == P) {
                // último atributo da linha é o target
                Y[i] = atof(token);
            } else {
                X[i][j] = atof(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }

    fclose(fp);

    




}
void combinacao_linear(double X[N][P],double B[P+1],double Z[N]){
    for(int i=0;i<N;i++){
        Z[i]=B[0];
        for(int j=0;j<P;j++){
            Z[i]+=((B[j+1]))*X[i][j];


        }
    }
}
void sigmoide(double p[N],double Z[N]){
    for(int i=0;i<N;i++){
        p[i]=1.0/(1.0+expf(-Z[i]));

    }
}
void gradiente(double gradJ_B[P+1],double p[N],double Y[N],double X[N][P]){
   double grad[P+1];
    for (int j = 0; j <= P; ++j) grad[j] = 0.0;

    for (int i = 0; i < N; ++i) {
        double err = p[i] - Y[i];
        grad[0] += err;                      // bias
        for (int j = 0; j < P; ++j)
            grad[j+1] += err * X[i][j];     // itera j contiguamente
    }

    for (int j = 0; j <= P; ++j)
        gradJ_B[j] = grad[j] / (double)N;
}

void atualizar_parametros(double B[P+1],double gradJ_B[P+1], double alfa){
    for( int j=0;j<P+1;j++){
        B[j]-=alfa*gradJ_B[j];
    }
}
void imprimir_parametros(double B[P+1]){
    for( int j=0;j<P+1;j++){
        printf("Parametro %d: %.2f\n", j,B[j]);
    }
}
void iniciar_dados_de_teste(double X[Nnew][P], double Y[Nnew]){
    
    FILE *fp = fopen("Cardiovascular_Disease_Dataset.csv", "r");
    if (fp == NULL) {
        perror("Erro ao abrir o arquivo");
       // return 1;
    }
    char linha[1024];
    int i = 0;

    // ler a primeira linha (cabeçalho) e descartar
    for(int i=0;i<N+10;i++)
    fgets(linha, sizeof(linha), fp);

    // ler até 1000 linhas ou até acabar arquivo
    while (i < Nnew && fgets(linha, sizeof(linha), fp)) {
        char *token;
        int j = 0;

        // pegar primeiro campo (id) e ignorar
        token = strtok(linha, ",");
        token = strtok(NULL, ","); // vai para o campo idade

        // processar atributos (idade até penúltimo campo)
        while (token != NULL && j <= P) {
            if (j == P) {
                // último atributo da linha é o target
                Y[i] = atof(token);
            } else {
                X[i][j] = atof(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }

    fclose(fp);
}
void imprimir_resultados(double X[Nnew][P],double B[P+1],double Y[Nnew]){
    printf("Resultados de teste:\n");
    int acertos=0;
     for(int i=0;i<Nnew;i++){
          double Zcalculado=B[0];
          for(int j=0;j<P;j++){
            Zcalculado+=B[j+1]*X[i][j];
          }
          double Ycalculado=1.0/(1.0+expf(-Zcalculado));
          printf("Ycalculado: %.2f Yreal: %.2f\n", Ycalculado,Y[i]);
          if(fabs(Ycalculado-Y[i])<0.1)
          acertos++;

     }
     printf("Acuracia: %.2f\n", (double)acertos/Nnew);

}

int main(){
 double X[N][P];
 double Y[N];
 double Z[N];
 double p[N];
 double B[P+1];
 double gradJ_B[P+1]={0};
 iniciar_parametros(B,0.1);//
 iniciar_dados_de_treino(X,Y);//
 struct timeval start, end;
 gettimeofday(&start, NULL);
 for (int i=0;i<500000;i++){//
    
    combinacao_linear(X,B,Z);
    sigmoide(p,Z);
    gradiente(gradJ_B,p,Y,X);
    double alfa=0.01;
    atualizar_parametros(B,gradJ_B,alfa);//
    
 }
gettimeofday(&end, NULL);
imprimir_parametros(B);//
double Xteste[Nnew][P];
double Yteste[Nnew];
iniciar_dados_de_teste(Xteste,Yteste);//
imprimir_resultados(Xteste,B,Yteste);//
long segundos = end.tv_sec - start.tv_sec;
long micros = end.tv_usec - start.tv_usec;
double tempo_total = segundos + micros / 1e6;
printf("Tempo: %.6f segundos\n", tempo_total);
return 0;
}