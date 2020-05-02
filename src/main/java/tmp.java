package it.unipd.dei.bdc1718;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.*;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.Scanner;


public class G06HM3 {

    // First of all, summarize the realized methods.

    /* Calculate the distance between a point x and a set of points S, as inf[d(x,s)], with s in S */
    private static double distanceFromSet(Vector x, ArrayList<Vector> S){
        double d = Double.POSITIVE_INFINITY;                        // initialize to infinity
        for(Vector s : S){
            double dist = Math.sqrt(Vectors.sqdist(x,s));           // take the square root of sqdist(a,b) that returns the squared L2-distance
            if(dist < d) d = dist;                                  // if it's lower, update it
        }
        return d;
    }


    // K-CENTER method
    private static ArrayList<Vector> kcenter(ArrayList<Vector> PS, int k){
        ArrayList<Vector> P = (ArrayList<Vector>) PS.clone();
        ArrayList<Vector> C = new ArrayList<>();

        C.add(P.remove((int)(Math.random()*P.size())));             // choose the first center as a random point of P

        for (int i = 2; i <= k ; i++){                              // find the other k-1 centers
            double max = 0;
            Vector c = P.get(0);
            for(Vector p : P){                                      // find the point c in P - C that maximizes d(c,C)
                double distance = distanceFromSet(p, C);
                if(distance > max){
                    max = distance;
                    c = p;
                }
            }
            C.add(P.remove(P.indexOf(c)));                          /* Update C adding the new center c. To obtain P - C,
                                                                       each c added in C is removed from P */
        }
        return C;
    }


    //K-MEANS++ method
    private static ArrayList<Vector> kmeansPP(ArrayList<Vector> PS, ArrayList<Long> WP, int k){
        ArrayList<Vector> P = (ArrayList<Vector>) PS.clone();
        ArrayList<Vector> C = new ArrayList<>();

        int ind = (int)(Math.random()*P.size());
        C.add(P.remove(ind));                                       /* the first center is a random point chosen from P
                                                                       with uniform probability */
        WP.remove(ind);

        for (int i = 2; i <= k ; i++){                              // find the other k - 1 centers
            double den = 0;
            /* compute the denominator of the probability: prob(j) = wp(j)*(dp(j))^2/(sum_{q non center} wq*(dq)^2),
                it is the same until the set P-S doesn't change (i.e. next iteration of for above) */
            for (int q = 0; q < P.size(); q++){
                den +=  WP.get(q)*(Math.pow(distanceFromSet(P.get(q), C) , 2));
            }
            double sum =  WP.get(0)*(Math.pow(distanceFromSet(P.get(0), C) , 2)) / den; /* initialize the lowest bound for the
                                                                                           new center selection */
            int r;
            double x = Math.random();                               // define a random number x into [0,1]S
            for(r = 1; r < P.size() - 1; r++){
                double num = WP.get(r)*(Math.pow(distanceFromSet(P.get(r), C) , 2));    // compute the numerator of probability
                if(sum <= x && x <= (sum + num/den)) break;         // find the only r such that sum_{j from 1 to r-1} prob(j) <= x <= sum_{j from 1 to r} prob(j)
                sum += num/den;                                     // if both the bound are not verified, try with follow r
            }

            C.add(P.remove(r));                                     /* Update C adding the new center c. To obtain P - C,
                                                                       each c added in C is removed from P */
            WP.remove(r);                                           // Remove also it's weight from WP

        }
        return C;
    }


    //K-MEANS OBJECTIVE FUNCTION
    private static double kmeansObj(ArrayList<Vector> P, ArrayList<Vector> C){
        double obj = 0;
        for (Vector p : P) {
            obj += Math.pow(distanceFromSet(p, C), 2);              // sum the squared distance of each point of P from
        }                                                           // its closest center (i.e. from the set C of center)

        return obj/P.size();                                        // returns the average squared distance
    }

    /*-------------------------------------------------MAIN---------------------------------------------------------------*/
    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        // INPUT: Allow to user to select which dataset and parameters to use

        /* The four datasets to test the program are passed on the command line in increasing order of size */
        Scanner in = new Scanner(System.in);
        System.out.println("Please, choose the size of the dataset to use: " +
                "\n [0] if you want 9960 points" +
                "\n [1] for 50047 points" +
                "\n [2] for 99670 points" +
                "\n [3] for 499950 points");
        int i = 0;
        try{
            i = in.nextInt();

            switch(i){
                case 0:
                    System.out.println("You have chosen: vecs-50-10000.txt \n");
                    break;
                case 1:
                    System.out.println("You have chosen: vecs-50-50000.txt \n");
                    break;
                case 2:
                    System.out.println("You have chosen: vecs-50-100000.txt \n");
                    break;
                case 3:
                    System.out.println("You have chosen: vecs-50-500000.txt \n");
                    break;
                default:
                    System.out.println("Please, choose one of the possibilities, not others \n");
            }
        }
        catch (InputMismatchException e){
            System.out.println("Wrong choice, the input must be an integer!!");
        }

        // Represent the chosen dataset as P, a set of points in Euclidean space
        ArrayList<Vector> P = InputOutput.readVectorsSeq(args[i]);

        /* Two integers k, k1 are received in input with the only conditions that they must be less than |P| and
        not equal between them.
        Then their assignment to the two variables, based on k < k1, is made by the program */
        int k = 0;
        int k1 = 0;
        Scanner inp = new Scanner(System.in);
        System.out.println("Please, enter the integers k, k1 (max " + P.size() + ") which are the two different sizes for the set of centers: ");
        try{
            int c = inp.nextInt();
            int c1 = inp.nextInt();

            if(c > P.size() || c1 > P.size())                // if both required conditions are not verify, ask to retry
                System.out.println("At least one is too big, retry.");
            else if(c == c1)                                 // even if equals, ask to retry
                System.out.println("They are the same, please enter two different integers.");
            else{
                if(c < c1){                                  // select the lowest one as k, the other one as k1
                    k = c;
                    k1 = c1;
                }
                else{                                        // select the lowest one as k, the other one as k1
                    k = c1;
                    k1 = c;
                }
                System.out.println("Well, you have set k = " + k + " and k1 = " + k1 + ".\n");
            }
        }
        catch (InputMismatchException e){                    // manage eventual wrong parameters
            System.out.println("Wrong choice, the inputs must be two integers!!");
        }
        in.close();
        inp.close();                                         // close both input flows

        // END INPUT.



// 1- Measure and print the k-center running time

        long start = System.currentTimeMillis();

        kcenter(P, k);                                       // find k centers in P by applying kcenter

        long end = System.currentTimeMillis();
        System.out.println("The running time of the kcenter algorithm is: " + (end - start) + " ms");


// 2- Obtain a set of k centers C with the k-means method and print the value return by kmeansObj

        ArrayList<Long> WP = new ArrayList<>();                     // build the list of weights
        for (int w = 0; w < P.size(); w++){
            WP.add(1L);                                             // set all weights in WP equal to 1
            //WP.add((long)(Math.random()*10));                     // or set them with e.g. random value in [0,10)
        }

        ArrayList<Vector> C = kmeansPP(P, WP, k);                   // Set of centers returned by kmeansPP, applied on P with weights WP

        System.out.println("The average squared distance of a point of P from its closest center, using only meansPP with "+k+ " centers, is: " + kmeansObj(P, C));


// 3- Test whether k1>k centers extracted with the kcenter primitive can provide a good coreset on which running kmeans++

        ArrayList<Vector> X = kcenter(P, k1);                       // obtain a set of k1 centers X by applying the kcenter method on P

        ArrayList<Long> WX = new ArrayList<>();                     // build the list of weights
        for (int w = 0; w < X.size(); w++){
            WX.add(1L);                                             // set all weights in WX equal to 1
            //WX.add((long)(Math.random()*10));                         // or set them with e.g. random value in [0,10)
        }

        ArrayList<Vector> CX = kmeansPP(X, WX, k);                  //obtain a set of k centers CX of X with kmeansPP method

        System.out.println("The average squared distance of a point of P from its closest center, using firstly kcenter and then kmeansPP, is: " + kmeansObj(P, CX));

        /* It is reasonable to see that increasing the values of k and k1, both approaches return a lower squared average distance
         (because better is the set of returned centers and the association of points to clusters). In particular, as expected the
         second approach returns worst results in all the cases (due the large approximation introduced by the application of the
         algorithm on a reduced set of points). The difference between the the first and second applications can be reduced by choosing
         an high value for k1 (i.e. trying with the first dataset and k=30, by passing from k1=80 to k1=1000, the difference between
         the second and the first average squared distances passes from values approximately equal to 5.5 , to 1.7).
        */

    }
}