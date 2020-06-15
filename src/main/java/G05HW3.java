import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.util.*;

public class G05HW3 {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter = 0; iter < k / 2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i + 1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD kcenter
    // Implementation of the Fastest-First-Traversal Algorithm
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    private static ArrayList<Vector> kcenter(ArrayList<Vector> P, int k) {
        ArrayList<Vector> C = new ArrayList<>();
        ArrayList<Double> dist = new ArrayList<>();             // arraylist of the distances. Simmetrical to P
        for (int i = 0; i < P.size(); i++) {
            dist.add(Double.POSITIVE_INFINITY);                 // initialize all the distances to infinity
        }

        C.add(P.get(0));                                        // choose the first center as a random point of P

        for (int i = 0; i < dist.size(); i++) {
            dist.set(i, Math.sqrt(Vectors.sqdist(C.get(0), P.get(i))));     // Update the distances from the first selected point
        }

        for (int l = 1; l < k; l++) {   //since we select all the k center do:
            int max_index = dist.indexOf(Collections.max(dist)); //choose the point at max distance from the center set
            C.add(P.get(max_index));                             //add to the center set

            //update the disctances in the following way: For each remaining not-yet-selected point q,
            // replace the distance stored for q by the minimum of its old value and the distance from p to q.
            for (int i = 0; i < P.size(); i++) {
                double d1 = dist.get(i);
                double d2 = Math.sqrt(Vectors.sqdist(C.get(l), P.get(i)));
                dist.set(i, Math.min(d1, d2));
            }

        }
        return C;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runMapReduce
    // Implementation for the Round 1 and 2.
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    private static ArrayList<Vector> runMapReduce(JavaRDD<Vector> inputPoints, int k, int L) {
        long start, end;

        // Measure the time taken by the coreset construction
        start = System.currentTimeMillis();

        JavaRDD<Vector> vecs = inputPoints
                .repartition(L)
                .mapPartitions((partition) -> {
                    // ArrayList to store the vectors inside the partition
                    ArrayList<Vector> vectors = new ArrayList<Vector>();

                    // Fill the ArrayList
                    while (partition.hasNext())
                        vectors.add(partition.next());

                    // Find the k centers using Farthest-First Traversal primitive
                    ArrayList<Vector> centers = kcenter(vectors, k);

                    // Iterator to store the k centers

                    // Return k centers
                    return centers.iterator();
                });

        long pointsNumber = vecs.count();       // used to avoid lazy evaluation
        assert (L * k == pointsNumber);         // check if we're using L*k points in coreset

        // Stop the timer
        end = System.currentTimeMillis();

        System.out.println("Runtime of Round 1 = " + (end - start) + " ms");

        // Measure the time taken by the computation of the final solution (through the sequential algorithm) on the coreset
        start = System.currentTimeMillis();

        // ArrayList to store the coreset
        ArrayList<Vector> coreset = new ArrayList<>(vecs.collect());

        // ArrayList to store the resulting k center
        ArrayList<Vector> result = runSequential(coreset, k);

        // Stop the timer
        end = System.currentTimeMillis();

        System.out.println("Runtime of Round 2 = " + (end - start) + " ms");

        return result;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD measure
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    public static double measure(ArrayList<Vector> pointsSet) {
        // ----------------------- COMPUTE THE SUM OF ALL PAIRWISE DISTANCES ---------------------

        // Initialize the numerator
        double sum = 0;
        int den = 0;

        // First point loop
        for (int i = 0; i < pointsSet.size(); i++) {
            // Take the first point
            Vector first = pointsSet.get(i);

            // Second point loop
            for (int j = 1 + i; j < pointsSet.size(); j++) {
                // Take the second point
                Vector second = pointsSet.get(j);

                // Compute the square distance between the first and the second point
                sum += Math.sqrt(Vectors.sqdist(first, second));
                den++;
            }
        }
        return sum / den;
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static void main(String[] args) throws IOException {

        if (args.length == 0)
            throw new IllegalArgumentException("Excepting the file name on the command line");

        // Setup Spark
        SparkConf conf = new SparkConf(true).setAppName("G05HW3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        //------- DISABLE LOG MESSAGES
        Logger.getLogger("org").setLevel(Level.OFF);

        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt((args[2]));

        long start = System.currentTimeMillis();

        JavaRDD<Vector> inputPoints = sc.textFile(args[0])
                .map(str -> strToVector(str))
                .cache();

        long size = inputPoints.count();   // used for avoid lazy evaluation of RDD

        long end = System.currentTimeMillis();

        System.out.println("Number of points = " + size);
        System.out.println("k = " + k);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + (end - start) + " ms");

        ArrayList<Vector> solution = runMapReduce(inputPoints, k, L);

        double avg = measure(solution);

        System.out.println("Average distance = " + avg);

    }
}