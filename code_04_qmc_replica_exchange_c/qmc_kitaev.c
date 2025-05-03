/*
 * Replica Exchange QMC for Kitaev model
 * C implementation with Fortran LAPACK diagonalization
 * Compile: gcc -O3 -std=c99 -I/usr/include -llapack -lblas -lm -o qmc_kitaev qmc_kitaev.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>

/* Fortran LAPACK prototype */
extern void zheev_(char *jobz, char *uplo, int *n,
                   double complex *a, int *lda,
                   double *w,
                   double complex *work, int *lwork,
                   double *rwork,
                   int *info);

/* site index */
static inline int site_id(int x, int y, int orb, int Lx) {
    return (y * Lx + x) * 2 + orb;
}

/* allocate bonds: returns count */
int make_bonds(int Lx, int Ly, int **bonds, double J[3]) {
    int maxb = Lx * Ly * 3;
    int *B = malloc(maxb * 3 * sizeof(int)); // i,j,type
    int cnt = 0;
    J[0]=1.0/3.0; J[1]=1.0/3.0; J[2]=1.0/3.0;
    for(int y=0;y<Ly;y++) for(int x=0;x<Lx;x++){
        int i=site_id(x,y,0,Lx), j=site_id(x,y,1,Lx);
        B[cnt*3+0]=i; B[cnt*3+1]=j; B[cnt*3+2]=0; cnt++;  /* x */
        if(x<Lx-1){
            i=site_id(x+1,y,0,Lx); j=site_id(x,y,1,Lx);
            B[cnt*3+0]=i; B[cnt*3+1]=j; B[cnt*3+2]=1; cnt++;  /* y */
        } else if(y<Ly-1){
            i=site_id(0,y+1,0,Lx); j=site_id(x,y,1,Lx);
            B[cnt*3+0]=i; B[cnt*3+1]=j; B[cnt*3+2]=1; cnt++;
        }
        i=site_id(x,y,0,Lx); j=site_id(x,(y+1)%Ly,1,Lx);
        B[cnt*3+0]=i; B[cnt*3+1]=j; B[cnt*3+2]=2; cnt++; /* z */
    }
    *bonds = B;
    return cnt;
}

/* build Hamiltonian H (column-major) */
void make_H(int N,int nbond,int*bonds,double J[3],int*eta, double complex*H){
    memset(H,0,N*N*sizeof(double complex));
    int cnt=0;
    for(int b=0;b<nbond;b++){
        int i=bonds[b*3], j=bonds[b*3+1], t=bonds[b*3+2];
        double JJ=J[t];
        if(t==2){ double s=eta[cnt++];
            H[j+i*N]= 2.0*JJ*I*s;
            H[i+j*N]=-2.0*JJ*I*s;
        } else {
            H[j+i*N]= 2.0*JJ*I;
            H[i+j*N]=-2.0*JJ*I;
        }
    }
}

/* diagonalize H in-place via zheev_ */
void diag(int N,double complex*H,double*w){
    char jobz='N', uplo='L';
    int n=N, lda=N, info;
    /* workspace query */
    int lwork=-1;
    double complex wkopt;
    double *rwork=malloc((3*N-2) * sizeof(double));
    zheev_(&jobz,&uplo,&n,H,&lda,w,&wkopt,&lwork,rwork,&info);
    lwork=(int)creal(wkopt);
    double complex *work=malloc(lwork*sizeof(double complex));
    zheev_(&jobz,&uplo,&n,H,&lda,w,work,&lwork,rwork,&info);
    if(info) { fprintf(stderr,"zheev_ error %d\n",info); exit(1); }
    free(work); free(rwork);
}

/* compute free energy metrics */
void free_energy(int N,double complex*H,double beta,double*F,double*E,double*dE){
    double *w=malloc(N*sizeof(double));
    diag(N,H,w);
    int half=N/2; double f=0,e=0, de=0;
    for(int i=0;i<half;i++){
        double eps=w[i], x=0.5*beta*eps;
        double ax=fabs(x), emx=exp(-ax), em2x=exp(-2*ax);
        f+=ax+log1p(em2x);
        e+= x * tanh(x);
        de+= x*x * pow(2.0*emx/(1.0+em2x),2);
    }
    *F=-f/beta; *E=-e/beta; *dE= -de/(beta*beta);
    free(w);
}

int main(int argc,char**argv){
    int L=2,nT=100,swap=1;
    double Tmin=0.01,Tmax=10.0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--L")) L=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--Tmin")) Tmin=atof(argv[++i]);
        else if(!strcmp(argv[i],"--Tmax")) Tmax=atof(argv[++i]);
        else if(!strcmp(argv[i],"--nT")) nT=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--swap")) swap=atoi(argv[++i]);
    }
    int Lx=L,Ly=L,R=nT;
    double *temps=malloc(R*sizeof(double)),*betas=malloc(R*sizeof(double));
    for(int i=0;i<R;i++){
        temps[i]=pow(10, log10(Tmin)+(double)(R-1-i)/(R-1)*(log10(Tmax)-log10(Tmin)));
        betas[i]=1.0/temps[i];
    }
    int *bonds, nbond = make_bonds(Lx,Ly,&bonds,(double[]){1./3,1./3,1./3});
    int nbondz=0; for(int b=0;b<nbond;b++) if(bonds[b*3+2]==2) nbondz++;
    int N=2*Lx*Ly;
    int **eta=malloc(R*sizeof(int*));
    int meas=0, Nth=10000,Nm=40000;
    for(int r=0;r<R;r++){ eta[r]=malloc(nbondz*sizeof(int)); for(int k=0;k<nbondz;k++) eta[r][k]=(rand()&1)?1:-1; }
    double *F=calloc(R,sizeof(double)),*E=calloc(R,sizeof(double)),*dE=calloc(R,sizeof(double));
    double **Es=malloc(R*sizeof(double*)),**dEs=malloc(R*sizeof(double*));
    for(int r=0;r<R;r++){ Es[r]=calloc(Nm,sizeof(double)); dEs[r]=calloc(Nm,sizeof(double)); }
    double complex *H=malloc(N*N*sizeof(double complex));
    for(int r=0;r<R;r++) free_energy(N,(make_H(N,nbond,bonds,(double[]){1./3,1./3,1./3},eta[r],H),H),betas[r],&F[r],&E[r],&dE[r]);
    for(int st=0;st<Nth+Nm;st++){
        if(st%swap==0){ for(int i=0;i<R-1;i++){ int j=i+1;
            double Fij,Fji,Eij,Eji,dEij,dEji;
            free_energy(N,(make_H(N,nbond,bonds,(double[]){1./3,1./3,1./3},eta[j],H),H),betas[i],&Fij,&Eij,&dEij);
            free_energy(N,(make_H(N,nbond,bonds,(double[]){1./3,1./3,1./3},eta[i],H),H),betas[j],&Fji,&Eji,&dEji);
            double expn=-betas[i]*Fij - betas[j]*Fji + betas[i]*F[i] + betas[j]*F[j];
            if(log((double)rand()/RAND_MAX) < fmin(0,expn)){
                int *te=eta[i]; eta[i]=eta[j]; eta[j]=te;
                F[i]=Fij;F[j]=Fji;
                E[i]=Eij;E[j]=Eji;
                dE[i]=dEij;dE[j]=dEji;
            }
        }}
        for(int r=0;r<R;r++){
            for(int k=0;k<N;k++){ int idx=rand()%nbondz; eta[r][idx]=-eta[r][idx];
                double Fn,En,dEn;
                free_energy(N,(make_H(N,nbond,bonds,(double[]){1./3,1./3,1./3},eta[r],H),H),betas[r],&Fn,&En,&dEn);
                double logP=-betas[r]*(Fn-F[r]);
                if(logP>=0||log((double)rand()/RAND_MAX)<logP) {F[r]=Fn;E[r]=En;dE[r]=dEn;} else eta[r][idx]=-eta[r][idx];
            }
        }
        if(st>=Nth){ for(int r=0;r<R;r++){ Es[r][meas]=E[r]; dEs[r][meas]=dE[r]; } meas++; }
    }
    FILE*fp=fopen("dat_specheat_rex","w");
    for(int r=0;r<R;r++){
        double Em=0,E2=0,dEm=0;
        for(int i=0;i<meas;i++){ Em+=Es[r][i]; E2+=Es[r][i]*Es[r][i]; dEm+=dEs[r][i]; }
        Em/=meas; E2/=meas; dEm/=meas;
        double Cv=betas[r]*betas[r]*(E2-Em*Em-dEm);
        fprintf(fp,"%e %e\n",temps[r],Cv/(2*Lx*Ly));
    }
    fclose(fp);
    return 0;
}

