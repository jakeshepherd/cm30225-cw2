#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

void printArray(double *arr, int dimension) {
   for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
         printf("%f, ", arr[dimension * i + j]);
      }
      printf("\n");
   }
}

double *readArrayFromFile(int argc, char *argv[], double *testValues, int currentRank, int *dimension, double *precision) {
    char inputFilename[32];

    if (argc >= 2) {
        *dimension = atoi(argv[1]);
    }
    if (argc >= 3) {
        *precision = strtod(argv[2], NULL);
    }
    if (argc >= 4) {
        strcpy(inputFilename, argv[3]);
    }

    if (currentRank == 0) {
        FILE *fp;
        char dimStr[6];
        fp = fopen(inputFilename, "r");
        if (fp == NULL) {
            perror("Error opening file");
            exit(-1);
        }

        if (fgets(dimStr, 6, fp) != NULL) {
            *dimension = atoi(dimStr);
        }
        if (*&dimension < 0) {
            printf("\n[ERROR] dimension cannot be zero.\n");
            exit(1);
        }

        unsigned long arrSize = sizeof(double) * (unsigned) (*dimension * *dimension);
        testValues = malloc(arrSize);

        for (int i=0; i<*dimension; i++) {
            for (int j=0; j<*dimension; j++) {
                char read[32];
                fscanf(fp, " %s", read);
                testValues[*dimension * i + j] = strtod(read, NULL);
            }
        }
        fclose(fp);
    }

    return testValues;
}

/*
 * Function: distributeRowIndexesToProccesors
 * ----------------------------
 * Allocate an even distribution of rows to available processors. The 
 * available processors is equal to the total processors - 1. If the 
 * problem size is not evenly divisable, the remainders will be spread
 * over the first n processors, with n equal to the remainder rows.
 * 
 * rowSplitPerProcessor: int array to store the result
 * dimension: dimension of the problem array
 * numOfProcessors: total processors being used
 */
void distributeRowIndexesToProccesors(int *rowSplitPerProcessor, int dimension, int numOfProcessors) {
    // accounting for no work in root
    int useableProcs = numOfProcessors - 1;

    rowSplitPerProcessor[0] = 0;
    for (int i = 1; i <= useableProcs; i++) {
        rowSplitPerProcessor[i] = (dimension - 2) / useableProcs;
    }

    int remainder = (dimension - 2) % useableProcs;
    if (remainder > 0) {
        // evenly distribute the remainder rows
        for (int i = 1; i <= remainder; i++) {
            rowSplitPerProcessor[i]++;
        }
    }
}

/*
 * Each processor will run this function individually on the data that it has to average it
 */
void calculateAverage(double *oldAverages, int numberOfRowsToAverage, int dimension, double prec, bool *processorPrecisionReached) {
    double *temp = malloc(sizeof(double) * (unsigned) (numberOfRowsToAverage * dimension));
    memcpy(temp, oldAverages, sizeof(double) * (unsigned) (numberOfRowsToAverage * dimension));
    *processorPrecisionReached = true;

    for (int i = 1; i < numberOfRowsToAverage - 1; i++) {
        for (int j = 1; j < dimension - 1; j++) {
            double average = (
                temp[dimension * (i + 1) + j] +
                temp[dimension * (i - 1) + j] +
                temp[dimension * i + (j + 1)] +
                temp[dimension * i + (j - 1)]
            ) / 4.0;

            if (fabs(temp[dimension * i + j] - average) > prec) {
                oldAverages[dimension * i + j] = average;
                *processorPrecisionReached = false;
            }
        }
    }

    free(temp);
}

/*
 * Function: testIt
 * ----------------------------
 * Runs the solver and iterates until the problem has converged.
 * 
 * testValues: starting array, will be an empty pointer in all processors except 
 *       the root
 * dimension: dimension of the problem array 
 * prec: precision at which the problem has converged 
 * currentRank: rank of the current processor 
 * numOfProcessors: total number of processors being used
 */
void testIt(double *testValues, int dimension, double prec, int currentRank, int numOfProcessors) {
    double startTime, runTime, averageRuntime;
    int rootProcessor = 0, numberOfIterations = 0, numberOfBoundaryRows = 2, numberOfRowsForProcessor, elementsForProcessor;
    double *updatedRows, *dataPerProcessor = NULL;
    bool precisionReached = false;

    if (currentRank == rootProcessor) {
        printf("INTIAL ARRAY\n");
        printArray(testValues, dimension);
    }

    // find number of rows to process per processor
    int *rowSplitPerProcessor = malloc(sizeof(int) * (unsigned) (numOfProcessors));
    distributeRowIndexesToProccesors(rowSplitPerProcessor, dimension, numOfProcessors);

    // Barrier here to make sure that all the processors available are all ready to go
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    do {
        if (currentRank == rootProcessor) {
            // If this is the first time going round
            // send every element of the data that needs to be processed
            if (numberOfIterations == 0) {
                int totalElementsSent = 0;

                for (int i = 1; i < numOfProcessors; i++) {
                    int elementsToSend = (rowSplitPerProcessor[i] + numberOfBoundaryRows) * dimension;

                    MPI_Send(&testValues[totalElementsSent], elementsToSend, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    totalElementsSent += elementsToSend - (numberOfBoundaryRows * dimension);
                }
            } 
            // else, only send the boundaryRowsPerProcessor of the data that needs to be processed
            else {
                int numberOfRowsSent = 0;
                double *boundaryRowsPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfBoundaryRows * dimension));

                // find the boundaries rows for each part of the data and send them to each processor
                for (int i = 1; i < numOfProcessors; i++) {
                    int upperBoundary = numberOfRowsSent;
                    int lowerBoundary = upperBoundary + rowSplitPerProcessor[i] + 1;

                    for (int j = 0; j < dimension; j++) {
                        boundaryRowsPerProcessor[dimension * 0 + j] = testValues[dimension * upperBoundary + j];
                        boundaryRowsPerProcessor[dimension * 1 + j] = testValues[dimension * lowerBoundary + j];
                    }

                    MPI_Send(boundaryRowsPerProcessor, (numberOfBoundaryRows * dimension), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    numberOfRowsSent += rowSplitPerProcessor[i];
                }
                free(boundaryRowsPerProcessor);
            }

            precisionReached = true;
            bool processorPrecisionReached = false;

            // Receive from all processors if they have reached precision
            // if they have not, then set precisionReached to false so that we can start the loop again later
            for (int i = 1; i < numOfProcessors; i++) {
                // receive bool indicating dataPerProcessor convergence from each child processor
                MPI_Recv(&processorPrecisionReached, 1, MPI_C_BOOL, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (!processorPrecisionReached) {
                    precisionReached = false;
                }
            }

            // Let all the processors know if the precision has been reached
            // This is so that they can stop processing if the precision has been reached
            MPI_Bcast(&precisionReached, 1, MPI_C_BOOL, rootProcessor, MPI_COMM_WORLD);

            // If the precision has been reached, we can start putting our array back together
            if (precisionReached) {
                int rowsReceivedTracker = 1;
                for (int i = 1; i < numOfProcessors; i++) {
                    numberOfRowsForProcessor = rowSplitPerProcessor[i] + numberOfBoundaryRows;
                    elementsForProcessor = numberOfRowsForProcessor * dimension;

                    updatedRows = malloc(sizeof(double) * (unsigned) elementsForProcessor);
                    MPI_Recv(updatedRows, elementsForProcessor, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Merge the data from the processor back into testValues
                    for (int i = 0; i < numberOfRowsForProcessor - numberOfBoundaryRows; i++) {
                        for (int j = 1; j < dimension - 1; j++) {
                            testValues[dimension * (rowsReceivedTracker + i) + j] = updatedRows[dimension * (i+1) + j];
                        }
                    }
                    rowsReceivedTracker += numberOfRowsForProcessor - numberOfBoundaryRows;
                    free(updatedRows);
                }
            } 
            // If the precision is not reached, we only need to merge the boundary rows into testValues
            // Because the processors are keeping a track of the other values
            else {
                int firstIndex, lastIndex = 0;
                double *boundaryRowsPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfBoundaryRows * dimension));

                for (int i = 1; i < numOfProcessors; i++) { 
                    MPI_Recv(boundaryRowsPerProcessor, (numberOfBoundaryRows * dimension), MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    firstIndex = lastIndex + 1;
                    lastIndex = firstIndex + rowSplitPerProcessor[i] - 1;

                    for (int j = 1; j < dimension-1; j++) {
                        testValues[firstIndex * dimension + j] = boundaryRowsPerProcessor[dimension * 0 + j];
                        testValues[lastIndex * dimension + j] = boundaryRowsPerProcessor[dimension * 1 + j];
                    }
                }

                free(boundaryRowsPerProcessor);
            }
        } else {
            // If it's the first iteration, we get a lot of data so handle separately
            if (numberOfIterations == 0) {
                numberOfRowsForProcessor = rowSplitPerProcessor[currentRank] + numberOfBoundaryRows;
                elementsForProcessor = numberOfRowsForProcessor * dimension;
                dataPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfRowsForProcessor * dimension));

                MPI_Recv(dataPerProcessor, elementsForProcessor, MPI_DOUBLE, rootProcessor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } 
            // On every other iteration, we only get the boundary rows
            else {
                double *boundaryRowsPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfBoundaryRows * dimension));
                MPI_Recv(boundaryRowsPerProcessor, (numberOfBoundaryRows * dimension), MPI_DOUBLE, rootProcessor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < dimension; i++) {
                    dataPerProcessor[i] = boundaryRowsPerProcessor[i];
                    dataPerProcessor[dimension * (numberOfRowsForProcessor - 1) + i] = boundaryRowsPerProcessor[dimension + i];
                }
                free(boundaryRowsPerProcessor);
            }

            bool processorPrecisionReached = false;
            calculateAverage(dataPerProcessor, numberOfRowsForProcessor, dimension, prec, &processorPrecisionReached);

            // Send whether our data has reached precision or not
            MPI_Send(&processorPrecisionReached, 1, MPI_C_BOOL, rootProcessor, 2, MPI_COMM_WORLD);

            // Receive the broadcast from root on the state of precisionReached
            MPI_Bcast(&precisionReached, 1, MPI_C_BOOL, rootProcessor, MPI_COMM_WORLD);

            // If precision reached, send our averaged data back to the root
            if (precisionReached) {
                MPI_Send(dataPerProcessor, elementsForProcessor, MPI_DOUBLE, rootProcessor, 1, MPI_COMM_WORLD);
            } 
            // If not reached, then we only want to send the boundary rows for the data back to update testValues
            else {
                double *boundaryRowsPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfBoundaryRows * dimension));

                for (int j = 0; j < dimension; j++) {
                    boundaryRowsPerProcessor[dimension * 0 + j] = dataPerProcessor[dimension * 1 + j];
                    boundaryRowsPerProcessor[dimension * 1 + j] = dataPerProcessor[dimension * (numberOfRowsForProcessor - numberOfBoundaryRows) + j];
                }

                MPI_Send(boundaryRowsPerProcessor, (numberOfBoundaryRows * dimension), MPI_DOUBLE, rootProcessor, 3, MPI_COMM_WORLD);
                free(boundaryRowsPerProcessor);
            }
        }
        numberOfIterations++;
    } while (!precisionReached);

    // Barrier here to make sure that every processor has finished it's work
    // This is so that we do not start calculating the runtime before the work has been done
    // We can then stop the timer and work out the runtime.
    MPI_Barrier(MPI_COMM_WORLD);
    runTime = MPI_Wtime() - startTime;

    // Use reduce to reduce (add) all the runtimes down into averageRuntime which is then later actually averaged
    MPI_Reduce(&runTime, &averageRuntime, 1, MPI_DOUBLE, MPI_SUM, rootProcessor, MPI_COMM_WORLD);

    if (currentRank == rootProcessor) {
        averageRuntime /= numOfProcessors;
        printf("-------------------\nRuntime: %f\n", averageRuntime);
        printf("\n\nFINAL ARRAY:\n");
        printArray(testValues, dimension);
    }

    if (dataPerProcessor != NULL) {
        free(dataPerProcessor);
    }
    free(rowSplitPerProcessor);
}

int main(int argc, char *argv[]) {
    int currentRank, numOfProcessors, dimension = 0;
    double precision = 0.01;
    double *testValues = NULL;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcessors);

    testValues = readArrayFromFile(argc, argv, testValues, currentRank, &dimension, &precision);

    testIt(testValues, dimension, precision, currentRank, numOfProcessors);
    MPI_Finalize();

    if (testValues != NULL) {
        free(testValues);
    }

    return 0;
}