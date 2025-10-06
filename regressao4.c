#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define NUM_FEATURES 49
#define NUM_ITER 10000  // número de iterações hardcoded
#define CSV_FILE "diabetes_dataset.csv"

// Função sigmoid estável numericamente
double sigmoid(double z) {
    if (z >= 0) {
        double ez = exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = exp(z);
        return ez / (1.0 + ez);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <num_samples> <num_threads> <run_id>\n", argv[0]);
        return 1;
    }

    int num_samples = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int run_id = atoi(argv[3]);
    omp_set_num_threads(num_threads);

    // --- Aloca memória ---
    float **X = (float **)malloc(num_samples * sizeof(float *));
    for (int i = 0; i < num_samples; i++)
        X[i] = (float *)calloc(NUM_FEATURES, sizeof(float));

    double *y = (double *)malloc(num_samples * sizeof(double));
    double *w = (double *)calloc(NUM_FEATURES, sizeof(double));
    double lr = 0.001;  // learning rate reduzido

    // Inicializa pesos pequenos
    for (int j = 0; j < NUM_FEATURES; j++)
        w[j] = ((double)rand()/RAND_MAX - 0.5) * 0.01;

    // --- Leitura do CSV ---
    FILE *fp = fopen(CSV_FILE, "r");
    if (!fp) {
        fprintf(stderr, "Erro: não foi possível abrir %s\n", CSV_FILE);
        return 1;
    }

    char line[2048];
    fgets(line, sizeof(line), fp); // Ignora cabeçalho

    for (int i = 0; i < num_samples && fgets(line, sizeof(line), fp); i++) {
        char gender[16], ethnicity[16], education[16], income[16];
        char employment[16], smoking[16], diabetes_stage[32];
        int patient_id, age, phys_act, diet_score, family_hist, hyper_hist;
        int cardio_hist, sys_bp, dia_bp, hr, diab_risk;
        float alcohol, sleep, screen, bmi, whr, chol, hdl, ldl, trig;
        float glucose_f, glucose_p, insulin, hba1c;
        int diagnosed;

        sscanf(line,
            "%d,%d,%15[^,],%15[^,],%15[^,],%15[^,],%15[^,],%15[^,],%f,%d,%d,%f,%f,%d,%d,%d,%f,%f,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%d,%31[^,],%d",
            &patient_id, &age, gender, ethnicity, education, income,
            employment, smoking, &alcohol, &phys_act, &diet_score,
            &sleep, &screen, &family_hist, &hyper_hist, &cardio_hist,
            &bmi, &whr, &sys_bp, &dia_bp, &hr, &chol, &hdl, &ldl, &trig,
            &glucose_f, &glucose_p, &insulin, &hba1c, &diab_risk, diabetes_stage, &diagnosed
        );

        int col = 0;
        X[i][col++] = age;
        // One-hot: gender
        X[i][col++] = strcmp(gender, "Male")==0;
        X[i][col++] = strcmp(gender, "Female")==0;
        X[i][col++] = strcmp(gender, "Other")==0;
        // One-hot: ethnicity
        X[i][col++] = strcmp(ethnicity, "White")==0;
        X[i][col++] = strcmp(ethnicity, "Hispanic")==0;
        X[i][col++] = strcmp(ethnicity, "Black")==0;
        X[i][col++] = strcmp(ethnicity, "Asian")==0;
        X[i][col++] = strcmp(ethnicity, "Other")==0;
        // One-hot: education
        X[i][col++] = strcmp(education, "No formal")==0;
        X[i][col++] = strcmp(education, "Highschool")==0;
        X[i][col++] = strcmp(education, "Graduate")==0;
        X[i][col++] = strcmp(education, "Postgraduate")==0;
        // Income
        X[i][col++] = strcmp(income, "Low")==0;
        X[i][col++] = strcmp(income, "Medium")==0;
        X[i][col++] = strcmp(income, "High")==0;
        // Employment
        X[i][col++] = strcmp(employment, "Employed")==0;
        X[i][col++] = strcmp(employment, "Unemployed")==0;
        X[i][col++] = strcmp(employment, "Retired")==0;
        X[i][col++] = strcmp(employment, "Student")==0;
        // Smoking
        X[i][col++] = strcmp(smoking, "Never")==0;
        X[i][col++] = strcmp(smoking, "Former")==0;
        X[i][col++] = strcmp(smoking, "Current")==0;
        // Numéricos
        X[i][col++] = alcohol;
        X[i][col++] = phys_act;
        X[i][col++] = diet_score;
        X[i][col++] = sleep;
        X[i][col++] = screen;
        X[i][col++] = family_hist;
        X[i][col++] = hyper_hist;
        X[i][col++] = cardio_hist;
        X[i][col++] = bmi;
        X[i][col++] = whr;
        X[i][col++] = sys_bp;
        X[i][col++] = dia_bp;
        X[i][col++] = hr;
        X[i][col++] = chol;
        X[i][col++] = hdl;
        X[i][col++] = ldl;
        X[i][col++] = trig;
        X[i][col++] = glucose_f;
        X[i][col++] = glucose_p;
        X[i][col++] = insulin;
        X[i][col++] = hba1c;
        X[i][col++] = diab_risk;

        y[i] = diagnosed;
    }
    fclose(fp);

    // --- Normalização ---
    for (int j = 0; j < NUM_FEATURES; j++) {
        float sum = 0.0f, sq = 0.0f;
        for (int i = 0; i < num_samples; i++) sum += X[i][j];
        float mean = sum / num_samples;
        for (int i = 0; i < num_samples; i++) sq += (X[i][j] - mean)*(X[i][j] - mean);
        float std = sqrtf(sq / num_samples);
        if (std < 1e-6f) std = 1.0f;
        for (int i = 0; i < num_samples; i++) X[i][j] = (X[i][j] - mean) / std;
    }

    double bias = 0.0;   // inicializa bias
    double dbias;         // gradiente do bias
    // --- Treinamento ---
    double start = omp_get_wtime();

    for (int iter = 0; iter < NUM_ITER; iter++) {
        double grad[NUM_FEATURES] = {0.0};
        dbias = 0.0;

        #pragma omp parallel for reduction(+:grad[:NUM_FEATURES], dbias)
        for (int i = 0; i < num_samples; i++) {
            double z = bias;   // soma do bias
            for (int j = 0; j < NUM_FEATURES; j++)
                z += w[j] * X[i][j];        // soma ponderada dos features
            double pred = sigmoid(z);       // função de ativação
            double err = pred - y[i];       // diferença predito - real

            for (int j = 0; j < NUM_FEATURES; j++)
                grad[j] += err * X[i][j];           // gradiente dos pesos

            dbias += err;  // gradiente do bias
        }

        for (int j = 0; j < NUM_FEATURES; j++)
            w[j] -= lr * grad[j] / num_samples;  // atualização dos pesos

        bias -= lr * dbias / num_samples;  // atualização do bias
    }

    double end = omp_get_wtime();

    // --- Cálculo da perda final ---
    double loss = 0.0;
    double z;
    for (int i = 0; i < num_samples; i++) {
        z = bias;
        for (int j = 0; j < NUM_FEATURES; j++)
            z += w[j] * X[i][j];
        double pred = sigmoid(z);
        if (pred < 1e-15) pred = 1e-15;
        if (pred > 1.0 - 1e-15) pred = 1.0 - 1e-15;
        loss += - (y[i]*log(pred) + (1-y[i])*log(1 - pred));
    }
    loss /= num_samples;

    // --- Saída CSV limpa ---
    printf("%d,%d,%d,%.6f,%.6f\n", num_threads, num_samples, run_id, end-start, loss);

    for (int i = 0; i < num_samples; i++) free(X[i]);
    free(X); free(y); free(w);
    return 0;
}
