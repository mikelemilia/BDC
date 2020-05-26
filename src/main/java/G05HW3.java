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

    private static double distanceFromSet(Vector x, ArrayList<Vector> S) {
        double d = Double.NEGATIVE_INFINITY;                        // initialize to infinity
        for (Vector s : S) {
            double dist = Math.sqrt(Vectors.sqdist(x, s));          // compute the squared root of a point from a point set
            if (dist > d) d = dist;
        }
        return d;
    }

    // Farthest-First Traversal
    private static ArrayList<Vector> kcenter(ArrayList<Vector> P, int k) {
        ArrayList<Vector> C = new ArrayList<>();

        C.add(P.remove((int) (Math.random() * P.size())));

        for (int i = 2; i <= k; i++) {
            double max = 0;
            Vector c = P.get(0);
            for (Vector p : P) {
                double distance = distanceFromSet(p, C);
                if (distance > max) {
                    max = distance;
                    c = p;
                }
            }
            C.add(P.remove(P.indexOf(c)));
        }
        return C;
    }

    // runMapReduce
    private static ArrayList<Vector> runMapReduce(JavaRDD<Vector> inputPoints, int k, int L) {
        long start, end;

        // Measure the time taken by the coreset construction
        start = System.currentTimeMillis();

        List<Vector> vecs = inputPoints.repartition(L)
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
                })
                .collect();

        // ArrayList to store the coreset
        ArrayList<Vector> coreset = new ArrayList<>(vecs);

        // Stop the timer
        end = System.currentTimeMillis();

        System.out.println("Runtime of Round 1 = " + (end - start) + "ms");

        // Measure the time taken by the computation of the final solution (through the sequential algorithm) on the coreset
        start = System.currentTimeMillis();

        // ArrayList to store the resulting k
        ArrayList<Vector> result = runSequential(coreset, k);

        // Stop the timer
        end = System.currentTimeMillis();

        System.out.println("Runtime of Round 2 = " + (end - start) + "ms");

        return result;
    }

    public static double measure(ArrayList<Vector> pointsSet) {
        // ----------------------- COMPUTE THE SUM OF ALL PAIRWISE DISTANCES ---------------------

        // Initialize the numerator
        double sum = 0;

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
            }

        }

        // ------------------------- COMPUTE THE NUMBER OF DISTINCT PAIRS -------------------------

//        // Initialize the denominator
//        double distinct = 0;
//
//        for (int k = 1; k < pointsSet.size(); k++) {
//            distinct += pointsSet.size() - k;
//        }

        // ------------- COMPUTE THE AVERAGE DISTANCE BETWEEN ALL POINTS IN pointslist -------------
//          double average = sum / distinct;

        return sum / ((pointsSet.size() * (pointsSet.size() - 1)) / 2); // TODO check
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
        Logger.getLogger("akka").setLevel(Level.OFF);


        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt((args[2]));

        long start = System.currentTimeMillis();

        JavaRDD<Vector> inputPoints = sc.textFile(args[0]/*, L*/)
                .map(str -> strToVector(str))
                .repartition(L)
                .cache();

        long end = System.currentTimeMillis();

        System.out.println("Number of points = " + inputPoints.count());
        System.out.println("k = " + k);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + (end - start) + "ms");

        ArrayList<Vector> solution = runMapReduce(inputPoints, k, L);

        double avg = measure(solution);

        System.out.println("Average distance = " + avg);

    }
}