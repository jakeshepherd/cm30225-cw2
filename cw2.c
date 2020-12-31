#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define ROOT_PROCESSOR_RANK 0
#define NUM_OF_BOUNDARY_ROWS 2

/*
 * Function: printArray
 * ----------------------------
 * Iterates through the given array and prints each element. 
 * Assumes the array is a square, 2D array.
 * 
 * vals: pointer to the array to print
 * dim: dim of the array
 */
void printArray(double *vals, int dim) {
    for (int i=0; i<dim; i++) {
        printf("(%d)\t", i);
        for (int j=0; j<dim; j++) {
            printf("%f, ", vals[dim * i + j]);
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
 * dim: dim of the problem array
 * precision: precision at which the array has converged
 * rank: rank of the processor
 * totalProcs: total processors being used
 * rowSplit: array detailing the number of rows per processor
 */
void printStartingInfo(int dim, double precision, int rank, int totalProcs, int *rowSplit) {
    if (rank == ROOT_PROCESSOR_RANK) {
        printf("\n\ndim: %d\tPrecision: %f\tProcessors: %d\n", dim, precision, totalProcs);
        printf("Work split: [");

        for (int i = 0; i < totalProcs; i++) {
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
 * dim: dim of the problem array
 * totalProcs: total processors being used
 */
void splitRowsPerProcessor(int *dest, int dim, int totalProcs) {
    int useableProcs = totalProcs - 1; // accounting for no work in root

    dest[0] = 0;
    for (int i = 1; i <= useableProcs; i++) {
        dest[i] = (dim - NUM_OF_BOUNDARY_ROWS) / useableProcs;
    }

    int remainder = (dim - NUM_OF_BOUNDARY_ROWS) % useableProcs;
    if (remainder > 0) {
        // evenly distribute the remainder rows
        for (int i = 1; i <= remainder; i++) {
            dest[i]++;
        }
    }
}

void averageRows(double *readArr, int numRows, int dim, double prec, bool *chunkConverged) {
    double *temp = malloc(sizeof(double) * (unsigned) (numRows * dim));
    memcpy(temp, readArr, sizeof(double) * (unsigned) (numRows * dim));
    *chunkConverged = true;

    for (int i = 1; i < numRows - 1; i++) {
        for (int j = 1; j < dim - 1; j++) {
            double avg = (
                temp[dim * (i + 1) + j] +
                temp[dim * (i - 1) + j] + 
                temp[dim * i + (j + 1)] + 
                temp[dim * i + (j - 1)]
                ) / 4.0;

            if (fabs(avg - temp[dim * i + j]) > prec) {
                readArr[dim * i + j] = avg;
                *chunkConverged = false;
            }
        }
    }

    free(temp);
}

/*
 * Function: relaxMpi
 * ----------------------------
 * Runs the solver and iterates until the problem has converged.
 * 
 * vals: starting array, will be an empty pointer in all processors except 
 *       the root
 * dim: dim of the problem array 
 * prec: precision at which the problem has converged 
 * currentRank: rank of the current processor 
 * totalProcs: total number of processors being used
 */
void relaxMpi(double *vals, int dim, double prec, int currentRank, int totalProcs) {
    double runtime, avgRuntime;
    int itCount = 0;
    bool hasConverged = false;

    // find number of rows to process per processor
    int *rowSplit = malloc(sizeof(int) * (unsigned) (totalProcs));
    splitRowsPerProcessor(rowSplit, dim, totalProcs);

    printStartingInfo(dim, prec, currentRank, totalProcs, rowSplit);

    int rowsInChunk;
    int elemsInChunk;
    double *updatedRows;
    double *chunk; // chunk of rows

    // make sure all nodes are ready to work before timing
    MPI_Barrier(MPI_COMM_WORLD);
    runtime = MPI_Wtime();

    do { // iterate until converged
        if (currentRank == ROOT_PROCESSOR_RANK) {
            if (itCount == 0) {
                // send chunks to each processor
                int totalElementsSent = 0;

                for (int i = 1; i < totalProcs; i++) {
                    int numRowsToSend = rowSplit[i] + NUM_OF_BOUNDARY_ROWS;
                    int elementsToSend = numRowsToSend * dim;

                    MPI_Send(&vals[totalElementsSent], elementsToSend, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    totalElementsSent += elementsToSend - (NUM_OF_BOUNDARY_ROWS * dim);
                }
            } else {
                // send boundaries
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dim));
                int rowsSent = 0; // the total rows covered by the sends

                for (int i = 1; i < totalProcs; i++) {
                    int firstBoundaryRow = rowsSent;
                    int lastBoundaryRow = firstBoundaryRow + rowSplit[i] + 1;

                    for (int j = 0; j < dim; j++) {
                        boundaries[dim * 0 + j] = vals[dim * firstBoundaryRow + j];
                        boundaries[dim * 1 + j] = vals[dim * lastBoundaryRow + j];
                    }

                    MPI_Send(boundaries, (NUM_OF_BOUNDARY_ROWS * dim), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    rowsSent += rowSplit[i];
                }
                free(boundaries);
            }
            hasConverged = true;
            bool chunkHasConverged = false;

            // receive chunks are converged? & updated hasConverged
            // hasConverged = true if all chunks are converged
            // check whether chunks have hit precision, if any single chunk hasn't, withinPrecision is false
            for (int i = 1; i < totalProcs; i++) {
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
                for (int i = 1; i < totalProcs; i++) {
                    rowsInChunk = rowSplit[i] + NUM_OF_BOUNDARY_ROWS;
                    // size of the chunk in terms of doubles, without the buffers
                    elemsInChunk = rowsInChunk * dim;

                    updatedRows = malloc(sizeof(double) * (unsigned) elemsInChunk);
                    MPI_Recv(updatedRows, elemsInChunk, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // merge the averaged chunk into the main array
                    for (int i = 0; i < rowsInChunk - NUM_OF_BOUNDARY_ROWS; i++) {
                        for (int j = 1; j < dim - 1; j++) {
                            vals[dim * (totalRowsReceived + i) + j] = updatedRows[dim * (i+1) + j];
                        }
                    }
                    totalRowsReceived += rowsInChunk - NUM_OF_BOUNDARY_ROWS;
                    free(updatedRows);
                }
            } else {
                // rec boundaries & merge to main array
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dim));
                int firstIndex, lastIndex = 0;

                for (int i = 1; i < totalProcs; i++) { 
                    MPI_Recv(boundaries, (NUM_OF_BOUNDARY_ROWS * dim), MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    firstIndex = lastIndex + 1;
                    lastIndex = firstIndex + rowSplit[i] - 1;

                    for (int j = 1; j < dim -1; j++) {
                        vals[firstIndex * dim + j] = boundaries[dim * 0 + j];
                        vals[lastIndex * dim + j] = boundaries[dim * 1 + j];
                    }

                }

                free(boundaries);
            }
        } else {
            if (itCount == 0) {
                // rec chunks and store
                rowsInChunk = rowSplit[currentRank] + NUM_OF_BOUNDARY_ROWS;
                elemsInChunk = rowsInChunk * dim;

                // memory for received chunk
                chunk = malloc(sizeof(double) * (unsigned) (rowsInChunk * dim));

                MPI_Recv(chunk, elemsInChunk, MPI_DOUBLE, ROOT_PROCESSOR_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // rec boundaries and store
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dim));
                MPI_Recv(boundaries, (NUM_OF_BOUNDARY_ROWS * dim), MPI_DOUBLE, ROOT_PROCESSOR_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //set first row of portion to first border
                for (int i = 0; i < dim; i++) {
                    chunk[i] = boundaries[i];
                    chunk[dim * (rowsInChunk - 1) + i] = boundaries[dim + i];
                }

                free(boundaries);
            }

            // avg chunks and check converged
            bool chunkConverged = false;
            averageRows(chunk, rowsInChunk, dim, prec, &chunkConverged);

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
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dim));

                for (int j = 0; j < dim; j++) {
                    boundaries[dim * 0 + j] = chunk[dim * 1 + j];
                    boundaries[dim * 1 + j] = chunk[dim * (rowsInChunk - NUM_OF_BOUNDARY_ROWS) + j];
                }

                MPI_Send(boundaries, (NUM_OF_BOUNDARY_ROWS * dim), MPI_DOUBLE, ROOT_PROCESSOR_RANK, 3, MPI_COMM_WORLD);
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
        avgRuntime /= totalProcs;
        printf("-------------------\nRuntime: %f\n", avgRuntime);
        printf("\n\nTook %d iterations.\n\n", itCount);
    }

    // if (currentRank == ROOT_PROCESSOR_RANK) {
    //     printf("\n\nFINAL ARRAY:\n");
    //     printArray(vals, dim);
    // }

    free(rowSplit);
    free(chunk);
}

int main(int argc, char *argv[]) {
    int currentRank, totalProcs;

    // defaults, will be updated
    int dim = 0;
    double precision = 0.01;
    char inputFilename[32];
    double *initVals = NULL;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);

    if (totalProcs < 2) {
        printf("-- There must be at least 2 processors.\n");
        exit(-1);
    }

    // process command args
    if (argc <= 1) {
        printf("\nNo arguments found. Something's gone wrong.\n");
        return -1;
    }
    if (argc >= 2) {
        dim = atoi(argv[1]);
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
        char dimStr[6]; // max dim = 99,999 + termination char
        fp = fopen(inputFilename, "r");
        if (fp == NULL) {
            perror("Error opening file");
            return -1;
        }

        // read in array dim 
        if (fgets(dimStr, 6, fp) != NULL) {
            dim = atoi(dimStr);
        }
        if (dim < 0) {
            printf("\n[ERROR] dim cannot be zero.\n");
            return 1;
        }

        // total size of the full problem array
        unsigned long arrSize = sizeof(double) * (unsigned) (dim * dim);
        initVals = malloc(arrSize);

        // read in input array
        for (int i=0; i<dim; i++) {
            for (int j=0; j<dim; j++) {
                char read[32];
                fscanf(fp, " %s", read);
                initVals[dim * i + j] = strtod(read, NULL);
            }
        }
        fclose(fp);
    }

    // run the solver
    relaxMpi(initVals, dim, precision, currentRank, totalProcs);

    MPI_Finalize();
    free(initVals);

    return 0;
}