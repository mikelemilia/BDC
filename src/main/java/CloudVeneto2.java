import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import java.io.IOException;
import java.lang.Math;
import java.util.*;

public class G06HM4 {
    // Initialize the timers
    static long start2, start3, end2, end3;

    /*
        Sequential approximation algorithm based on matching
    */

    private static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {
        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            double d = Math.sqrt(Vectors.sqdist(points.get(i), points.get(j)));
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
    }

    // Calculate the distance between a point x and a set of points S, as inf[d(x,s)], with s in S
    private static double distanceFromSet(Vector x, ArrayList<Vector> S)
    {
        double d = Double.POSITIVE_INFINITY;
        for(Vector s : S)
        {
            double dist = Math.sqrt(Vectors.sqdist(x,s));
            if(dist < d) d = dist;
        }
        return d;
    }

    // Farthest-First Traversal
    private static ArrayList<Vector> kcenter(ArrayList<Vector> P, int k){
        ArrayList<Vector> C = new ArrayList<>();

        C.add(P.remove((int)(Math.random()*P.size())));

        for (int i = 2; i <= k ; i++)
        {
            double max = 0;
            Vector c = P.get(0);
            for(Vector p : P)
            {
                double distance = distanceFromSet(p, C);
                if(distance > max){
                    max = distance;
                    c = p;
                }
            }
            C.add(P.remove(P.indexOf(c)));
        }
        return C;
    }

    // runMapReduce
    private static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsrdd, int k, int numBlocks){
        // Measure the time taken by the coreset construction
        start2 = System.currentTimeMillis();

        List<Vector> vecs = pointsrdd.repartition(numBlocks)
                .mapPartitions( (partition) -> {
                    // ArrayList to store the vectors inside the partition
                    ArrayList<Vector> vectors = new ArrayList<Vector>();

                    // Fill the ArrayList
                    while(partition.hasNext())
                        vectors.add(partition.next());

                    // Find the k centers using Farthest-First Traversal primitive
                    ArrayList<Vector> centers = kcenter(vectors, k);

                    // Iterator to store the k centers
                    Iterator<Vector> centers_collection = centers.iterator();

                    // Return k centers
                    return centers_collection;
                })
                .collect();

        // ArrayList to store the coreset
        ArrayList<Vector> coreset = new ArrayList<Vector>(vecs);

        // Stop the timer
        end2 = System.currentTimeMillis();

        // Measure the time taken by the computation of the final solution (through the sequential algorithm) on the coreset
        start3 = System.currentTimeMillis();

        // ArrayList to store the resulting k
        ArrayList<Vector> result = runSequential(coreset, k);

        // Stop the timer
        end3 = System.currentTimeMillis();

        return result;
    }

    public static double measure(ArrayList<Vector> pointslist)
    {
        // ----------------------- COMPUTE THE SUM OF ALL PAIRWISE DISTANCES ---------------------

        // Initialize the numerator
        double sum = 0;

        // First point loop
        for(int i = 0; i < pointslist.size(); i++)
        {
            // Take the first point
            Vector first = pointslist.get(i);

            // Counter used to increase the value of j everytime i increases
            int count = 0;

            // Second point loop
            for (int j = 1 + count; j < pointslist.size(); j++)
            {
                // Take the second point
                Vector second = pointslist.get(j);

                // Compute the square distance between the first and the second point
                sum += Math.sqrt(Vectors.sqdist(first, second));
            }

            // Update the counter
            count++;
        }

        // ------------------------- COMPUTE THE NUMBER OF DISTINCT PAIRS -------------------------

        // Initialize the denominator
        double distinct = 0;

        for(int k = 1; k < pointslist.size(); k++)
        {
            distinct += pointslist.size() - k;
        }

        // ------------- COMPUTE THE AVERAGE DISTANCE BETWEEN ALL POINTS IN pointslist -------------
        double average = sum/distinct;

        return average;
    }

    public static void main (String[] args) throws IOException {
        if (args.length == 0)
            throw new IllegalArgumentException("Excepting the file name on the command line");

        // Input
        Scanner in = new Scanner(System.in);

        System.out.println("Choose k");
        int k = in.nextInt();
        System.out.println("Choose numBlocks");
        int numBlocks = in.nextInt();

        // Close input
        in.close();

        // Setup Spark
        SparkConf conf = new SparkConf(true).setAppName("Homework 4");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //Read the input points
        JavaRDD<Vector> pointsrdd = sc
                .textFile(args[0])
                .map(InputOutput::strToVector)
                .repartition(numBlocks)
                .cache();

        // Determine the solution of the max-diversity problem
        ArrayList<Vector> pointslist = runMapReduce(pointsrdd,k,numBlocks);

        // Average distance among the solution points
        double avg = measure(pointslist);

        // 1
        System.out.println("The average distance among the solution points with k = " + k + " is: " + avg);

        // 2
        System.out.println("The time taken by the coreset construction is: " + (end2 - start2) + " ms");

        // 3
        System.out.println("The time taken by the computation of the final solution (through the sequential algorithm) on the coreset is: " + (end3 - start3) + " ms");
    }
}