#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define ROOT_PROCESSOR_RANK 0
#define NUM_OF_BOUNDARY_ROWS 2

void printArray(double *arr, int dimension) {
   for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
         printf("%f, ", arr[dimension * i + j]);
      }
      printf("\n");
   }
}

/*
 * Function: printStartingInfo
 * ----------------------------
 * Simple function that prints out the starting values such as dimenstion
 * and the row split between processors
 * 
 * dimension: dimension of the problem array
 * precision: precision at which the array has converged
 * rank: rank of the processor
 * numOfProcessors: total processors being used
 * rowSplit: array detailing the number of rows per processor
 */
void printStartingInfo(int dimension, double precision, int rank, int numOfProcessors, int *rowSplit) {
    if (rank == ROOT_PROCESSOR_RANK) {
        printf("\n\ndim: %d\tPrecision: %f\tProcessors: %d\n", dimension, precision, numOfProcessors);
        printf("Work split: [");

        for (int i = 0; i < numOfProcessors; i++) {
            printf("p%d: %d, ", i, rowSplit[i]);
        }
        printf("]\n\n");
    }
}

/*
 * Function: splitRowsPerProcessor
 * ----------------------------
 * Allocate an even distribution of rows to available processors. The 
 * available processors is equal to the total processors - 1. If the 
 * problem size is not evenly divisable, the remainders will be spread
 * over the first n processors, with n equal to the remainder rows.
 * 
 * dest: int array to store the result
 * dimension: dimension of the problem array
 * numOfProcessors: total processors being used
 */
void splitRowsPerProcessor(int *dest, int dimension, int numOfProcessors) {
    int useableProcs = numOfProcessors - 1; // accounting for no work in root

    dest[0] = 0;
    for (int i = 1; i <= useableProcs; i++) {
        dest[i] = (dimension - NUM_OF_BOUNDARY_ROWS) / useableProcs;
    }

    int remainder = (dimension - NUM_OF_BOUNDARY_ROWS) % useableProcs;
    if (remainder > 0) {
        // evenly distribute the remainder rows
        for (int i = 1; i <= remainder; i++) {
            dest[i]++;
        }
    }
}

void averageRows(double *readArr, int numRows, int dimension, double prec, bool *chunkConverged) {
    double *temp = malloc(sizeof(double) * (unsigned) (numRows * dimension));
    memcpy(temp, readArr, sizeof(double) * (unsigned) (numRows * dimension));
    *chunkConverged = true;

    for (int i = 1; i < numRows - 1; i++) {
        for (int j = 1; j < dimension - 1; j++) {
            double avg = (
                temp[dimension * (i + 1) + j] +
                temp[dimension * (i - 1) + j] + 
                temp[dimension * i + (j + 1)] + 
                temp[dimension * i + (j - 1)]
                ) / 4.0;

            if (fabs(avg - temp[dimension * i + j]) > prec) {
                readArr[dimension * i + j] = avg;
                *chunkConverged = false;
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
    double runtime, avgRuntime;
    int itCount = 0;
    bool hasConverged = false;

    // find number of rows to process per processor
    int *rowSplit = malloc(sizeof(int) * (unsigned) (numOfProcessors));
    splitRowsPerProcessor(rowSplit, dimension, numOfProcessors);

    printStartingInfo(dimension, prec, currentRank, numOfProcessors, rowSplit);

    int rowsInChunk;
    int elemsInChunk;
    double *updatedRows;
    double *chunk = NULL; // chunk of rows

    // make sure all nodes are ready to work before timing
    MPI_Barrier(MPI_COMM_WORLD);
    runtime = MPI_Wtime();

    do { // iterate until converged
        if (currentRank == ROOT_PROCESSOR_RANK) {
            if (itCount == 0) {
                // send chunks to each processor
                int totalElementsSent = 0;

                for (int i = 1; i < numOfProcessors; i++) {
                    int numRowsToSend = rowSplit[i] + NUM_OF_BOUNDARY_ROWS;
                    int elementsToSend = numRowsToSend * dimension;

                    MPI_Send(&testValues[totalElementsSent], elementsToSend, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    totalElementsSent += elementsToSend - (NUM_OF_BOUNDARY_ROWS * dimension);
                }
            } else {
                // send boundaries
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dimension));
                int rowsSent = 0; // the total rows covered by the sends

                for (int i = 1; i < numOfProcessors; i++) {
                    int firstBoundaryRow = rowsSent;
                    int lastBoundaryRow = firstBoundaryRow + rowSplit[i] + 1;

                    for (int j = 0; j < dimension; j++) {
                        boundaries[dimension * 0 + j] = testValues[dimension * firstBoundaryRow + j];
                        boundaries[dimension * 1 + j] = testValues[dimension * lastBoundaryRow + j];
                    }

                    MPI_Send(boundaries, (NUM_OF_BOUNDARY_ROWS * dimension), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    rowsSent += rowSplit[i];
                }
                free(boundaries);
            }
            hasConverged = true;
            bool chunkHasConverged = false;

            // receive chunks are converged? & updated hasConverged
            // hasConverged = true if all chunks are converged
            // check whether chunks have hit precision, if any single chunk hasn't, withinPrecision is false
            for (int i = 1; i < numOfProcessors; i++) {
                // receive bool indicating chunk convergence from each child processor
                MPI_Recv(&chunkHasConverged, 1, MPI_C_BOOL, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (!chunkHasConverged) {
                    hasConverged = false;
                }
            }

            // broadcast array convergence to child processors
            MPI_Bcast(&hasConverged, 1, MPI_C_BOOL, ROOT_PROCESSOR_RANK, MPI_COMM_WORLD);

            if (hasConverged) {
                // rec chunks & merge to main array
                int totalRowsReceived= 1;
                for (int i = 1; i < numOfProcessors; i++) {
                    rowsInChunk = rowSplit[i] + NUM_OF_BOUNDARY_ROWS;
                    // size of the chunk in terms of doubles, without the buffers
                    elemsInChunk = rowsInChunk * dimension;

                    updatedRows = malloc(sizeof(double) * (unsigned) elemsInChunk);
                    MPI_Recv(updatedRows, elemsInChunk, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // merge the averaged chunk into the main array
                    for (int i = 0; i < rowsInChunk - NUM_OF_BOUNDARY_ROWS; i++) {
                        for (int j = 1; j < dimension - 1; j++) {
                            testValues[dimension * (totalRowsReceived + i) + j] = updatedRows[dimension * (i+1) + j];
                        }
                    }
                    totalRowsReceived += rowsInChunk - NUM_OF_BOUNDARY_ROWS;
                    free(updatedRows);
                }
            } else {
                // rec boundaries & merge to main array
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dimension));
                int firstIndex, lastIndex = 0;

                for (int i = 1; i < numOfProcessors; i++) { 
                    MPI_Recv(boundaries, (NUM_OF_BOUNDARY_ROWS * dimension), MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    firstIndex = lastIndex + 1;
                    lastIndex = firstIndex + rowSplit[i] - 1;

                    for (int j = 1; j < dimension -1; j++) {
                        testValues[firstIndex * dimension + j] = boundaries[dimension * 0 + j];
                        testValues[lastIndex * dimension + j] = boundaries[dimension * 1 + j];
                    }

                }

                free(boundaries);
            }
        } else {
            if (itCount == 0) {
                // rec chunks and store
                rowsInChunk = rowSplit[currentRank] + NUM_OF_BOUNDARY_ROWS;
                elemsInChunk = rowsInChunk * dimension;

                // memory for received chunk
                chunk = malloc(sizeof(double) * (unsigned) (rowsInChunk * dimension));

                MPI_Recv(chunk, elemsInChunk, MPI_DOUBLE, ROOT_PROCESSOR_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // rec boundaries and store
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dimension));
                MPI_Recv(boundaries, (NUM_OF_BOUNDARY_ROWS * dimension), MPI_DOUBLE, ROOT_PROCESSOR_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //set first row of portion to first border
                for (int i = 0; i < dimension; i++) {
                    chunk[i] = boundaries[i];
                    chunk[dimension * (rowsInChunk - 1) + i] = boundaries[dimension + i];
                }

                free(boundaries);
            }

            // avg chunks and check converged
            bool chunkConverged = false;
            averageRows(chunk, rowsInChunk, dimension, prec, &chunkConverged);

            // send chunk converged to root
            MPI_Send(&chunkConverged, 1, MPI_C_BOOL, ROOT_PROCESSOR_RANK, 2, MPI_COMM_WORLD);

            // receive broadcast for (all array) hasConverged
            // receive withinPrecision bool from the master process, is a blocking call so also acts as a synchronise
            MPI_Bcast(&hasConverged, 1, MPI_C_BOOL, ROOT_PROCESSOR_RANK, MPI_COMM_WORLD);

            if (hasConverged) {
                // send updated chunks to root
                // send back to the master process only modified inner rows of the chunk

                MPI_Send(chunk, elemsInChunk, MPI_DOUBLE, ROOT_PROCESSOR_RANK, 1, MPI_COMM_WORLD);
            } else {
                // send boundaries to root
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dimension));

                for (int j = 0; j < dimension; j++) {
                    boundaries[dimension * 0 + j] = chunk[dimension * 1 + j];
                    boundaries[dimension * 1 + j] = chunk[dimension * (rowsInChunk - NUM_OF_BOUNDARY_ROWS) + j];
                }

                MPI_Send(boundaries, (NUM_OF_BOUNDARY_ROWS * dimension), MPI_DOUBLE, ROOT_PROCESSOR_RANK, 3, MPI_COMM_WORLD);
                free(boundaries);
            }
        }

        printf("it: %d\n", itCount);
        itCount++;
    } while (!hasConverged);

    // and make sure all nodes have finished the work
    MPI_Barrier(MPI_COMM_WORLD);
    runtime = MPI_Wtime() - runtime;

    // reduce (sum) all processor runtimes into a single value in the root
    MPI_Reduce(&runtime, &avgRuntime, 1, MPI_DOUBLE, MPI_SUM, ROOT_PROCESSOR_RANK, MPI_COMM_WORLD);

    // and get the average runtime
    if (currentRank == ROOT_PROCESSOR_RANK) {
        avgRuntime /= numOfProcessors;
        printf("-------------------\nRuntime: %f\n", avgRuntime);
        printf("\n\nTook %d iterations.\n\n", itCount);
    }

    // if (currentRank == ROOT_PROCESSOR_RANK) {
    //     printf("\n\nFINAL ARRAY:\n");
    //     printArray(testValues, dimension);
    // }

    free(rowSplit);
    if (chunk != NULL) {
        free(chunk);
    }
}

int main(int argc, char *argv[]) {
    int currentRank, numOfProcessors;

    // defaults, will be updated
    int dimension = 0;
    double precision = 0.01;
    char inputFilename[32];
    double *testValues = NULL;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcessors);

    if (totalPro < 2) {
        printf("-- There must be at least 2 processors.\n");
        exit(-1);
    }

    // process command args
    if (argc <= 1) {
        printf("\nNo arguments found. Something's gone wrong.\n");
        return -1;
    }
    if (argc >= 2) {
        dimension = atoi(argv[1]);
    }
    if (argc >= 3) {
        precision = strtod(argv[2], NULL);
    }
    if (argc >= 4) {
        strcpy(inputFilename, argv[3]);
    }

    // read in data in root
    if (currentRank == ROOT_PROCESSOR_RANK) {
        FILE *fp;
        char dimStr[6]; // max dimension = 99,999 + termination char
        fp = fopen(inputFilename, "r");
        if (fp == NULL) {
            perror("Error opening file");
            return -1;
        }

        // read in array dim 
        if (fgets(dimStr, 6, fp) != NULL) {
            dimension = atoi(dimStr);
        }
        if (dimension < 0) {
            printf("\n[ERROR] dimension cannot be zero.\n");
            return 1;
        }

        // total size of the full problem array
        unsigned long arrSize = sizeof(double) * (unsigned) (dimension * dimension);
        testValues = malloc(arrSize);

        // read in input array
        for (int i=0; i<dimension; i++) {
            for (int j=0; j<dimension; j++) {
                char read[32];
                fscanf(fp, " %s", read);
                testValues[dimension * i + j] = strtod(read, NULL);
            }
        }
        fclose(fp);
    }

    // run the solver
    testIt(testValues, dimension, precision, currentRank, numOfProcessors);

    MPI_Finalize();
    if (testValues != NULL) {
        free(testValues);
    }

    return 0;
}