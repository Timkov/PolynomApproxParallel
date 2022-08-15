#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "plplot/plstream.h"
#include "mpi.h"

using namespace std;

enum ProcessRank { MAIN_PROC, POLYNOMS_CALC_PROC, TARGET_FUNCTION_CALC_PROC };
const int REQUIRED_PROCESSES_COUNT = 3;

const double STEP = 0.05;
const double MAX_PARAM = M_PI;
const double MAX_PARAM_SQ = MAX_PARAM * MAX_PARAM;
const double DELTA = 1e-3;
const int MAX_DEGREE = 15;
const int NO_VALUE = -2147483647;
const int POINTS_COUNT = int(trunc(MAX_PARAM / STEP));

int CHEB_COEFFS[MAX_DEGREE][MAX_DEGREE / 2 + 1] = {
	{      1,      0,      0,      0,      0,      0,      0,      0 },
	{      2,     -1,      0,      0,      0,      0,      0,      0 },
	{      4,     -3,      0,      0,      0,      0,      0,      0 },
	{      8,     -8,      1,      0,      0,      0,      0,      0 },
	{     16,    -20,      5,      0,      0,      0,      0,      0 },
	{     32,    -48,     18,     -1,      0,      0,      0,      0 },
	{     64,   -112,     56,     -7,      0,      0,      0,      0 },
	{    128,   -256,    160,    -32,      1,      0,      0,      0 },
	{    256,   -576,    432,   -120,      9,      0,      0,      0 },
	{    512,  -1280,   1120,   -400,     50,     -1,      0,      0 },
	{   1024,  -2816,   2816,  -1232,    220,    -11,      0,      0 },
	{   2048,  -6144,   6912,  -3584,    840,    -72,      1,      0 },
	{   4096, -13312,  16640,  -9984,   2912,   -364,     13,      0 },
	{   8192, -28672,  39424, -26880,   9408,  -1568,     98,     -1 },
	{  16384, -61440,  92160, -70400,   2880,  -6048,    560,    -15 },
};

double targetFunction(double x) {
	return pow(x * sin(x), 0.5);		// function to approximate
}

double conv(double x) {
	return sqrt(1 - pow((2 * x - MAX_PARAM) / MAX_PARAM, 2)) * MAX_PARAM_SQ;
}

double calcCheb(double x, int st) {
	double xst;
	int nst;
	double sum;
	int n;
	if (st == 0) {
		return 1;
	}
	x = (2 * x - MAX_PARAM) / MAX_PARAM;
	sum = 0;
	nst = st % 2;
	if (nst == 0) {
		xst = 1;
	}
	else {
		xst = x;
	}
	n = (st + 2) / 2;
	while (nst <= st) {
		sum = sum + CHEB_COEFFS[st - 1][n - 1] * xst;
		nst += 2;
		xst = xst * x * x;
		n--;
	}
	return sum;
}

int valToIdx(double val) {
	return int(round(val / STEP));
}

double calcRelativeError(double* target, double* approximated, int length) {
	double error = 0;
	int i;
	for (i = 0; i < length; ++i) {
		error += fabs((target[i] - approximated[i]) / target[i]);
	}
	return error / length;
}

void drawPlots(int n, double* x, double* y1, double* y2) {
	PLFLT* xP = x, * yP1 = y1, * yP2 = y2;
	PLFLT xmin = 0, xmax = x[n - 1] * 1.1, ymin = 0, ymax = NO_VALUE;
	PLINT dotCharCode = 17;
	int i;

											// calculate y min and max values
	for (i = 0; i < n; i++)
	{
		ymin = ymin > yP2[i] ? yP2[i] : ymin;
		ymax = ymax < yP2[i] ? yP2[i] : ymax;
		ymin = ymin > yP1[i] ? yP1[i] : ymin;
		ymax = ymax < yP1[i] ? yP1[i] : ymax;
	}
	ymax *= 1.1;

											// initialize plplot
	plscolbg(225, 225, 225);
	plinit();

											// create a labelled box to hold the plot.

	plcol0(DarkRed);
	plenv(xmin, xmax, ymin, ymax, 0, 0);
	pllab("x", "y=(x*sin(x))#u0.5#d", "Target and approximated functions");
	plcol0(Grey);
	plbox("uwg", 0.1, 0, "uvg", 0.1, 0);
	plcol0(DarkRed);
	plaxes(0, 0, "abct", 0.5, 0, "abct", 0.5, 0);
	plcol0(Red);
	plmtex("rv", -23, 0.12, 0, "Target:         _________");
	plcol0(DeepBlue);
	plmtex("rv", -23, 0.07, 0, "Approximated:  _________");

											// plot the data
	plcol0(Red);
	plpoin(n, xP, yP1, dotCharCode);
	plline(n, xP, yP1);
	plcol0(DeepBlue);
	plline(n, xP, yP2);
	plpoin(n, xP, yP2, dotCharCode);

											// close PLplot library
	plend();
}

int main(int* argc, char** argv) {

	int processRankId;
	ProcessRank processRank;
	int processesCount;
	MPI_Comm comm;
	MPI_Request sendRequest1, sendRequest2;

	int i, tempIdx, degree;
	double x, curApproximation;
	double* aCoefs = NULL;
	double* approxValues = NULL;
	double* targetValues = NULL;
	double* xArgs = NULL;
	double** chebPolynoms = NULL;

	MPI_Init(argc, &argv);
	comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &processRankId);
	MPI_Comm_size(comm, &processesCount);

	processRank = (ProcessRank)processRankId;

	if (processesCount < REQUIRED_PROCESSES_COUNT) {
		printf("Error: %d processes required to run the program, but %d provided", REQUIRED_PROCESSES_COUNT, processesCount);
		MPI_Finalize();
		return 0;
	}
											// allocate memory (only for processes which use it)
	
	if (processRank == MAIN_PROC) {
		aCoefs = (double*)malloc((MAX_DEGREE + 1) * sizeof(double));
		xArgs = (double*)malloc((POINTS_COUNT + 1) * sizeof(double));
		approxValues = (double*)malloc((2 * POINTS_COUNT) * sizeof(double));
	}
	if (processRank == TARGET_FUNCTION_CALC_PROC || processRank == MAIN_PROC) {
		targetValues = (double*)malloc((2 * POINTS_COUNT) * sizeof(double));
		for (x = 0; x <= MAX_PARAM; x += STEP) {
			targetValues[valToIdx(x)] = NO_VALUE;
		}
	}
	if (processRank == POLYNOMS_CALC_PROC || processRank == MAIN_PROC) {
		chebPolynoms = (double**)malloc((POINTS_COUNT + 1) * sizeof(double*));
		for (x = 0; x <= MAX_PARAM; x += STEP) {
			chebPolynoms[valToIdx(x)] = (double*)malloc((MAX_DEGREE + 1) * sizeof(double));
			for (int i = 0; i < MAX_DEGREE; ++i) {
				chebPolynoms[valToIdx(x)][i] = NO_VALUE;
			}
		}
	}

	MPI_Barrier(comm);

	if (processRank == TARGET_FUNCTION_CALC_PROC) {		// calc target function values
		for (x = 0; x <= MAX_PARAM; x += STEP) {
			tempIdx = valToIdx(x);
			targetValues[tempIdx] = targetFunction(x);
			MPI_Isend(									// unblocking send function value
				&targetValues[tempIdx],
				1,
				MPI_DOUBLE,
				MAIN_PROC,
				tempIdx,
				comm,
				&sendRequest1);
		}
	}
	else if (processRank == POLYNOMS_CALC_PROC) {		// calc polynoms
		for (i = 0; i < MAX_DEGREE; ++i) {
			for (x = 0; x <= MAX_PARAM; x += STEP) {
				tempIdx = valToIdx(x);
				chebPolynoms[tempIdx][i] = calcCheb(x, i);
				MPI_Isend(								// unblocking send polynom value
					&chebPolynoms[tempIdx][i],
					1,
					MPI_DOUBLE,
					MAIN_PROC,
					tempIdx * MAX_DEGREE + i,			// id in x12 
					comm,
					&sendRequest2);
			}
		}
	}
	else if (processRank == MAIN_PROC) {

		printf("Enter the maximum polynom degree:\n");
		fflush(stdout);
		scanf("%d", &degree);
		degree = degree > MAX_DEGREE ? MAX_DEGREE : (degree < 3 ? 3 : degree);

														// calc coefs
		aCoefs[0] = 0;
		x = STEP;
		while (x < MAX_PARAM - DELTA) {
			tempIdx = valToIdx(x);
			MPI_Recv(									// recieve function value
				&targetValues[tempIdx],
				1,
				MPI_DOUBLE,
				TARGET_FUNCTION_CALC_PROC,
				tempIdx,
				comm,
				MPI_STATUS_IGNORE);
			MPI_Recv(									// recieve polynom value for degree = 0
				&chebPolynoms[tempIdx][0],
				1,
				MPI_DOUBLE,
				POLYNOMS_CALC_PROC,
				tempIdx * MAX_DEGREE + 0,
				comm,
				MPI_STATUS_IGNORE);

			aCoefs[0] = aCoefs[0] + targetValues[valToIdx(x)] * STEP * 2 / conv(x);
			x = x + STEP;
		};
		for (i = 1; i < degree; i++) {
			aCoefs[i] = 0;
			x = STEP;

			while (x < MAX_PARAM - DELTA) {
				tempIdx = valToIdx(x);
				MPI_Recv(								// recieve polynom value
					&chebPolynoms[tempIdx][i],
					1,
					MPI_DOUBLE,
					POLYNOMS_CALC_PROC,
					tempIdx * MAX_DEGREE + i,
					comm,
					MPI_STATUS_IGNORE);

				aCoefs[i] = aCoefs[i] + targetValues[tempIdx] * chebPolynoms[tempIdx][i] * STEP * 4 / conv(x);
				x = x + STEP;
			}
		}
														// calc approx values

		x = STEP;
		while (x < MAX_PARAM - DELTA) {
			curApproximation = 0;
			tempIdx = valToIdx(x);
			for (i = 0; i < degree; i++) {						
				curApproximation = curApproximation + aCoefs[i] * chebPolynoms[tempIdx][i];		
			}

			approxValues[tempIdx] = curApproximation;

			xArgs[valToIdx(x)] = x;
			x = x + STEP;
		}
														// calc and print relative error

		printf("Relative error = %.3lf%%\n", calcRelativeError(targetValues + 1, approxValues + 1, int((MAX_PARAM - STEP) / STEP)) * 100);
		printf("___________________________________________________________________________");
		fflush(stdout);
														// draw plot

		drawPlots(int((MAX_PARAM - STEP) / STEP), xArgs + 1, targetValues + 1, approxValues + 1);		
	}

	MPI_Barrier(comm);

														// free memory
	free(aCoefs);
	free(approxValues);
	free(targetValues);
	free(xArgs);
	for (x = 0; x <= MAX_PARAM; x += STEP) {
		free(chebPolynoms[valToIdx(x)]);
	}
	free(chebPolynoms);


	MPI_Finalize();

	return 0;
}
