//
//  main.c
//  neural_net
//
//  Created by TU on 2016/09/27.
//  Copyright © 2016年 TU. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* 記号定数の定義 */
#define INPUTNO 3
#define HIDDENNO 3
#define ALPHA 10;   /* learning coefficient*/
#define SEED 65535;
#define MAXINPUTNO 100
#define BIGNUM 100      /* initial value of error */
#define LIMIT 0.001     /* upper limit of error */

/* 関数のプロトタイプ宣言 */
double f(double u);     /* 伝達関数 */
void initwh(double wh[HIDDENNO][INPUTNO + 1]);
void initwo(double wo[HIDDENNO+1]);
double drnd(void);                              /* generation of randomizer */
void print(double wh[HIDDENNO][INPUTNO +1],
           double wo[HIDDENNO +1]);             /* print result */
double forward(double wh[HIDDENNO][INPUTNO +1],
               double wo[HIDDENNO +1], double hi[],
               double e[INPUTNO + 1]);
void olearn(double wo[HIDDENNO+1], double hi[],
            double e[INPUTNO +1], double o);        /* adjustment of weights in output */
int getdata(double e[][INPUTNO]);
void hlearn(double wh[HIDDENNO][INPUTNO +1],
            double wo[HIDDENNO +1], double hi[],
            double e[INPUTNO +1], double o) ;       /* adjustment of weights in hidden */



/*************************
    main()関数
 ************************/

int main(){
    double wh[HIDDENNO][INPUTNO +1]; /* in hidden*/
    double wo[HIDDENNO +1];             /* in output*/
    double e[MAXINPUTNO][INPUTNO + 1];     /* dataset*/
    double hi[HIDDENNO +1];             /* output in hidden layer */
    double o;                           /* output */
    double err = BIGNUM;
    int i, j;                           /* control loop */
    int n_of_e;                         /* data number */
    int count = 0;
    
    /* nitialization of random seed */
    srand(SEED);
    
    
    /* initialization of weights */
    initwh(wh);
    initwo(wo);
    print(wh, wo);
    
    /* read input data */
    n_of_e = getdata(e);
    printf("data number: %d\n", n_of_e);
    
    /* learning */
    while (err > LIMIT){
        err = 0.0;
        for(j=0; j<n_of_e; ++j){
            /* forward direction */
            o =forward(wh, wo, hi, e[j]);
            /* weight adjust in output layer */
            olearn(wo, hi, e[j], o);
            /* weight adjust in hidden layer */
            hlearn(wh, wo, hi, e[j], o);
            /* calculate error */
            err += (o - e[j][INPUTNO]) * (o- e[j][INPUTNO]);
        }
        ++count;
        /* output error */
        fprintf(stderr, "%d\t%lf\n", count, err);
    } /* finish learning */
    
    /* output of concat weight */
    print(wh, wo);
    
    /* output for learning data */
    for (i=0; i < n_of_e; ++i){
        printf("%d ", i);
        for(j=0; j< INPUTNO +1; ++j)
            printf("%lf ", e[i][j]);
        o= forward(wh, wo, hi, e[i]);
        printf("%lf\n", o);
        
    }
    
    
    /* body of calculation */
    
    for(i=0; i< n_of_e; ++i){
        printf("%d ", i);
        for(j=0; j < INPUTNO; ++j)
            printf("%lf ", e[i][j]);
        printf("%lf\n", o);
    }
    
    return 0;
}

