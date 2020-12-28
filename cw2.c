#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

double *readArray;
double *writeArray;
   /* This program sums all rows in an array using MPI parallelism.
    * The root process acts as a master and sends a portion of the
    * array to each child process.  Master and child processes then
    * all calculate a partial sum of the portion of the array assigned
    * to them, and the child processes send their partial sums to 
    * the master, who calculates a grand total.
    **/
   
void populateArrayData(int dimension) {
    writeArray = malloc(sizeof(double) * (unsigned) (dimension * dimension));
    srand((unsigned int) time(NULL));

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            writeArray[dimension * i + j] = (double) rand()/(float) (RAND_MAX/3);
        }
    }
}

void printArray(double *arr, int dimension) {
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            printf("%f, ", arr[dimension * i + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
   int dimension = 6, numberOfThreads = 16;
   double precision = 0.01;

   MPI_Status status;
   int my_id, i, num_procs, num_rows_to_receive, sender;
   double average, partial_average;
   
   /* Now replicte this process to create parallel processes.
    * From this point on, every process executes a seperate copy
    * of this program */
   int ierr = MPI_Init(NULL, NULL);
   
   int root_process = 0;
   int sendCount = 0;
   
   /* find out MY process ID, and how many processes were started. */
   ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   if (my_id == root_process) {
      double *averageArray = malloc(sizeof(double) * (3 * (unsigned) (dimension)));
      // populate array
      populateArrayData(dimension);
      printArray(writeArray, dimension);
      printf("\n");

      // now go through array and set up the rows that will be sent
      double *toSend = malloc(sizeof(double) * (3 * (unsigned) (dimension)));

      // take one row and get it ready to send
      // this is kind of dodgy rn
      for (int i = 0; i<dimension; i++) {
         // int i = 0;
         for (int j = 0; j<dimension; j++) {
            toSend[dimension * i + j] = writeArray[dimension * (i) + j];
            toSend[dimension * (i+1) + j] = writeArray[dimension * (i+1) + j];
            toSend[dimension * (i+2) + j] = writeArray[dimension * (i+2) + j];
         }

         // send array to other processor(s)
         MPI_Send(toSend, 3*dimension, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
         sendCount++;

         printf("arrays sent from root.\n");

         printf("waiting for arrays on root\n");
         // need to collect it all together and combine into a new array that then gets sent back out
         
         // MPI_Recv(averageArray, 3*dimension, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
         // printf("Returned averages from processor %i\n", status.MPI_SOURCE);
      }

      
      free(averageArray);
      free(toSend);

      printf("\nDONE ON ROOT\n");
   } else {
      printf("ON THREAD\n");
      
      // receive sent data
      double *toAverage = malloc(sizeof(double) * (unsigned) (dimension)  * (unsigned) (dimension));

      // for (int i = 0; i<dimension; i++) {
         printf("waiting for arrays in processor... \n");
         ierr = MPI_Recv(toAverage, 3*dimension, MPI_DOUBLE, root_process, 1, MPI_COMM_WORLD, &status);
         printf("received arrays from root... \n");

         // array now has all the data we need in it
         // process the data that has been sent

         double *averageArray = malloc(sizeof(double) * (unsigned) (dimension));
         // fill up an average array to send back to root
         for (int x = 0; x<3; x++) {
            for(int j = 0; j<dimension; j++) {

               //todo this is wrong again
               averageArray[j] = (
                  toAverage[dimension * (x-1) + (j)] + 
                  toAverage[dimension * (x) + (j-1)] + 
                  toAverage[dimension * (x) + (j+1)] + 
                  toAverage[dimension * (x+1) + j]
               )/4;
            }
         }
      
      // }
      printf("\ntrying to send\n");
      /* and finally, send my average array back to the root process */
      MPI_Send(averageArray, 3*dimension, MPI_DOUBLE, root_process, 1, MPI_COMM_WORLD);
      printf("\n sent \n");
      free(toAverage);
      free(averageArray);

      printf("\nDONE ON PROCESSOR\n");
   }
   ierr = MPI_Finalize();
}