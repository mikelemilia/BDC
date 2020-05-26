import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import static org.apache.spark.mllib.linalg.BLAS.axpy;
import static org.apache.spark.mllib.linalg.BLAS.scal;

public class CloudVeneto {
    public static void main(String[] args) throws Exception {

        //------- PARSING CMD LINE ------------
        // Parameters are:
        // <path to file>, k, L and iter

        if (args.length != 4) {
            System.err.println("USAGE: <filepath> k L iter");
            System.exit(1);
        }
        String inputPath = args[0];
        int k = 0, L = 0, iter = 0;
        try {
            k = Integer.parseInt(args[1]);
            L = Integer.parseInt(args[2]);
            iter = Integer.parseInt(args[3]);
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (k <= 2 && L <= 1 && iter <= 0) {
            System.err.println("Something wrong here...!");
            System.exit(1);
        }
        //------------------------------------
        final int k_fin = k;

        //------- DISABLE LOG MESSAGES
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        //------- SETTING THE SPARK CONTEXT
        SparkConf conf = new SparkConf(true).setAppName("kmedian new approach");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //------- PARSING INPUT FILE ------------
        JavaRDD<Vector> pointset = sc.textFile(args[0], L)
                .map(x -> strToVector(x))
                .repartition(L)
                .cache();
        long N = pointset.count();
        System.out.println("Number of points is : " + N);
        System.out.println("Number of clusters is : " + k);
        System.out.println("Number of parts is : " + L);
        System.out.println("Number of iterations is : " + iter);

        //------- SOLVING THE PROBLEM ------------
        double obj = MR_kmedian(pointset, k, L, iter);
        System.out.println("Objective function is : <" + obj + ">");
    }

    public static Double MR_kmedian(JavaRDD<Vector> pointset, int k, int L, int iter) {
        //
        // --- ADD INSTRUCTIONS TO TAKE AND PRINT TIMES OF ROUNDS 1, 2 and 3
        //

        //------------- ROUND 1 ---------------------------

        long start = System.currentTimeMillis();

        JavaRDD<Tuple2<Vector, Long>> coreset = pointset.mapPartitions(x ->
        {
            ArrayList<Vector> points = new ArrayList<>();
            ArrayList<Long> weights = new ArrayList<>();
            while (x.hasNext()) {
                points.add(x.next());
                weights.add(1L);
            }
            ArrayList<Vector> centers = kmeansPP(points, weights, k, iter);
            ArrayList<Long> weight_centers = compute_weights(points, centers);
            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); ++i) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weight_centers.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        }).cache();

        coreset.count();

        long end = System.currentTimeMillis();

        System.out.println("Elapsed time for Round 1 (T1): " + (end - start) + " ms.");

        //------------- ROUND 2 ---------------------------

        start = System.currentTimeMillis();

        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>(k * L);
        elems.addAll(coreset.collect());
        ArrayList<Vector> coresetPoints = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();
        for (int i = 0; i < elems.size(); ++i) {
            coresetPoints.add(i, elems.get(i)._1);
            weights.add(i, elems.get(i)._2);
        }

        ArrayList<Vector> centers = kmeansPP(coresetPoints, weights, k, iter); // k final centers

        end = System.currentTimeMillis();

        System.out.println("Elapsed time for Round 2 (T2): " + (end - start) + " ms.");

        //------------- ROUND 3: COMPUTE OBJ FUNCTION --------------------

        start = System.currentTimeMillis();

        JavaRDD<Double> distances = pointset
                .mapPartitions(x -> {
                    ArrayList<Double> dist = new ArrayList<>();
                    while (x.hasNext()) {
                        org.apache.spark.mllib.linalg.Vector point = x.next();
                        double tmp = euclidean(centers.get(0), point);

                        // For every center in C
                        for (int j = 1; j < centers.size(); j++) {
                            // Min between current distance and the one calculated for the center j
                            tmp = Math.min(euclidean(centers.get(j), point), tmp);
                        }
                        dist.add(tmp);
                    }
                    return dist.iterator();
                });

        double obj = (distances.reduce((a, b) -> a + b)) / distances.count(); //distances and pointset have the same size

        end = System.currentTimeMillis();
        System.out.println("Elapsed time for Round 3 (T3): " + (end - start) + " ms.");

        return obj;
    }

    public static ArrayList<Long> compute_weights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for (int i = 0; i < points.size(); ++i) {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j) {
                if (euclidean(points.get(i), centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // Euclidean distance
    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }


    //kmeans++
    public static ArrayList<org.apache.spark.mllib.linalg.Vector> kmeansPP(ArrayList<org.apache.spark.mllib.linalg.Vector> P, ArrayList<Long> WP, int k, int iter) {
        // KMEANS++

        ArrayList<org.apache.spark.mllib.linalg.Vector> centers = new ArrayList<>();
        // I need at every iteration to maintain for each point in P its distance from the closest, among the current centers
        // ArrayList wDistances will contain distances already multiplied by weights
        ArrayList<Double> wDistances = new ArrayList<>(Collections.nCopies(P.size(), Double.MAX_VALUE));

        // Add to the set of center a random point chosen from P with uniform probability
        Random rand = new Random();
        centers.add(P.get(rand.nextInt(P.size())));

        // Add (k-1) other centers
        for (int i = 0; i < k - 1; i++) {
            // ArrayList with the probability to chose each point
            ArrayList<Double> probab = new ArrayList<>();
            // Variable to compute the sum of the distances (to compute the probability)
            double sum = 0;

            // Compute the distances
            for (int j = 0; j < P.size(); j++) {
                // Weighted distance between the center i and the point j in P
                double dist = Math.sqrt(Vectors.sqdist(centers.get(i), P.get(j))) * WP.get(j);

                // Keep min{computed weighted distance, old distance of the point j from the set of centers}
                wDistances.set(j, Math.min(dist, wDistances.get(j)));

                // Update the sum (sum also the centers, but their distances are 0)
                sum = sum + wDistances.get(j);
            }

            // Compute probability to chose each point as new center
            for (int j = 0; j < P.size(); j++) {
                // For each point : (his weighted distance from the set of centers)/sum
                probab.add(wDistances.get(j) / sum);
            }

            // Choose a point to add to the set of centers

            // Uniformly random value between [0,1]
            double randValue = rand.nextDouble();
            double probabSum = 0;
            boolean stopCondition = false;
            // Index of the chosen point
            int index = 0;

            while (index < P.size() && !stopCondition) {
                // Ignore points with distance = 0 from the set of centers (they are centers)
                if (wDistances.get(index) != 0) {
                    probabSum += probab.get(index);
                    if (probabSum > randValue) {
                        stopCondition = true;
                    }
                }
                index++;
            }
            centers.add(P.get(index - 1));
        }


        // WEIGHTED LLOYD

        // ArrayList partition will contain for each point in P the index of its center in centers
        ArrayList<Integer> partition;

        int t = 0;
        // ArrayList centersOpt will contain centers corresponding to the best objective function values, those that will be returned
        ArrayList<org.apache.spark.mllib.linalg.Vector> centersOpt = centers;

        // Objective function values
        double fiNew = 0;
        double fiBest = Double.MAX_VALUE;

        while (t < iter) {
            // Compute Partition(P, centers)
            partition = partition(P, centers);

            // For every cluster, update the center
            for (int i = 0; i < k; i++) {
                // Variable sum will contain the sum of the weights of vectors in the cluster, used to calculate the new center
                double sum = 0;
                // Vector temp, initialized with all zeroes, will contain the sum of vectors (in the cluster)
                // multiplied by their weights, used to calculate the new center
                org.apache.spark.mllib.linalg.Vector temp;
                double zeros[] = new double[(centers.get(0)).size()];
                Arrays.fill(zeros, 0);
                temp = Vectors.dense(zeros);

                for (int j = 0; j < P.size(); j++) {
                    // Points in the same cluster
                    if (partition.get(j) == i) {
                        sum += WP.get(j);
                        axpy(WP.get(j), P.get(j), temp);
                    }
                }

                scal((1 / sum), temp);
                centers.set(i, temp);
                // Now I have the new centers
            }

            // Check if the new objective function is better than the current best one: if so update the value and the corresponding best centers
            fiNew = kmedianFi(P, centers, WP, partition);

            if (fiNew < fiBest) {
                fiBest = fiNew;
                centersOpt = centers;
            }

            t++;
        }

        // Return, after performing iter iterations, the best objective function's value found and the corresponding best centers
        return centersOpt;
    }


    // Additional method to compute Partition primitive (used in Lloyd's iterations)

    public static ArrayList<Integer> partition(ArrayList<org.apache.spark.mllib.linalg.Vector> P, ArrayList<org.apache.spark.mllib.linalg.Vector> C) {
        // ArrayList partition will contain for each point in P the index of its center (saved in C)
        ArrayList<Integer> partition = new ArrayList<>();

        // For every point i in P
        for (int i = 0; i < P.size(); i++) {
            // Contains the corresponding center of each point
            int center = 0;

            // Contains the distance between each point and its center (initialized with the first center and then updated)
            double distanceOld = Math.sqrt(Vectors.sqdist(P.get(i), C.get(0)));

            // For every center in C
            for (int j = 1; j < C.size(); j++) {
                double distance = Math.sqrt(Vectors.sqdist(P.get(i), C.get(j)));

                // Keep min{computed new distance, old distance of the point i from the set of centers}
                if (distance < distanceOld) {
                    // Update closest distance and center
                    distanceOld = distance;
                    center = j;
                }
            }

            // Add for each point its closest center
            partition.add(center);
        }

        return partition;
    }


    // Additional method to compute the value of the objective function (used in Lloyd's iterations)

    public static double kmedianFi(ArrayList<org.apache.spark.mllib.linalg.Vector> P, ArrayList<org.apache.spark.mllib.linalg.Vector> C, ArrayList<Long> WP, ArrayList<Integer> partition) {
        double kmedianFi = 0;

        // For every center in C
        for (int i = 0; i < C.size(); i++) {
            // For every point in P
            for (int j = 0; j < P.size(); j++) {
                // If the center of point j is center i then sum the weighted distance
                if (partition.get(j) == i) {
                    kmedianFi += Math.sqrt(Vectors.sqdist(C.get(i), P.get(j))) * WP.get(j);
                }
            }
        }
        return kmedianFi;
    }


}
