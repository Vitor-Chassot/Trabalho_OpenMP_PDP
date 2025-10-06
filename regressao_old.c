#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define N 900
#define P 12
#define Nnew 70
void iniciar_dados(float X[N][P], float Y[N]){
    
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
void combinacao_linear(float X[N][P],float B[P+1],float Z[N]){
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        Z[i]=B[0];
        for(int j=0;j<P;j++){
            Z[i]+=((B[j+1]))*X[i][j];


        }
    }
}
void sigmoide(float p[N],float Z[N]){
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        p[i]=1.0/(1.0+expf(-Z[i]));

    }
}
void gradiente(float gradJ_B[P+1],float p[N],float Y[N],float X[N][P]){
    gradJ_B[0]=0.0;
    long double grad=0.0;
    #pragma omp parallel for reduction(+:grad)
    for(int i=0;i<N;i++){
        grad+=(p[i]-Y[i])*1;
        /*if(grad>1000000000){
                gradJ_B[0]+=grad/N;
                grad=0.0;
            }*/
    }
    gradJ_B[0]=grad/N;
    for(int j=0;j<P;j++){
        gradJ_B[j+1]=0.0;
        grad=0.0;
        #pragma omp parallel for reduction(+:grad)
        for(int i=0;i<N;i++){
            
        
            grad+=(p[i]-Y[i])*X[i][j];
            //printf("gradAcumlando: %Lf\n", grad);
            /*if(grad>100000000){
               //usleep(1000000);
                gradJ_B[j+1]+=grad/N;
                grad=0.0;
            }*/
            
        }
        gradJ_B[j+1]=grad/N;
        //printf("grad: %.2f\n", gradJ_B[j+1]);
    }
}
void atualizar_parametros(float B[P+1],float gradJ_B[P+1], float alfa){
    for( int j=0;j<P+1;j++){
        B[j]-=alfa*gradJ_B[j];
    }
}
void imprimir_parametros(float B[P+1]){
    for( int j=0;j<P+1;j++){
        printf("Parametro %d: %.2f\n", j,B[j]);
    }
}
void imprimir_resultados(float X[N][P],float B[P+1],float Y[N]){
     for(int i=0;i<N;i++){
          float Ycalculado=B[0];
          for(int j=0;j<P;j++){
            Ycalculado+=B[j+1]*X[i][j];
          }
          printf("Ycalculado: %.2f Yreal: %.2f\n", Ycalculado,Y[i]);

     }

}
void imprimir_novos_resultados(float X[Nnew][P],float B[P+1],float Y[Nnew]){
    int acertos=0;
     for(int i=0;i<Nnew;i++){
          float Zcalculado=B[0];
          for(int j=0;j<P;j++){
            Zcalculado+=B[j+1]*X[i][j];
          }
          float Ycalculado=1.0/(1.0+expf(-Zcalculado));
          //printf("Ycalculado: %.2f Yreal: %.2f\n", Ycalculado,Y[i]);
          int pred = (Ycalculado >= 0.5) ? 1 : 0;
          if(pred == (int)Y[i])
            acertos++;

     }
     printf("Acuracia: %.2f\n", (double)acertos/Nnew);

}
void testar_dados_novos(float X[Nnew][P], float Y[Nnew]){
    
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
        while (token != NULL && j < P) {
            X[i][j] = atof(token);
            j++;
            token = strtok(NULL, ",");
        }
        // último atributo da linha é o target
        Y[i] = atof(token);
        i++;
    }

    fclose(fp);

    




}


int main(){
 float X[N][P];
 float Y[N];
 float Z[N];
 float p[N];
 float B[P+1];
 for(int i=0;i<P+1;i++)
 B[i]=0.1;
 float gradJ_B[P+1]={0};
 iniciar_dados(X,Y);
double start, end;
start = omp_get_wtime();

 for (int i=0;i<100000;i++){
    
    combinacao_linear(X,B,Z);
    sigmoide(p,Z);
    gradiente(gradJ_B,p,Y,X);
    float alfa=0.01;
    atualizar_parametros(B,gradJ_B,alfa);
    
 }
 end = omp_get_wtime();
imprimir_parametros(B);
//imprimir_resultados(X,B,Y);
float Xteste[Nnew][P];
float Yteste[Nnew];
testar_dados_novos(Xteste,Yteste);
printf("Resultados de teste:\n");
imprimir_novos_resultados(Xteste,B,Yteste);
printf("Tempo de execucao (treinamento): %.3f segundos\n", end - start);
return 0;
}