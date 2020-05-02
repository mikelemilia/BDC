import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.ArrayList;
import java.util.Random;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class G05HW2 {

    private static double distanceFromSet(Vector x, ArrayList<Vector> S) {
        double d = Double.NEGATIVE_INFINITY;                        // initialize to infinity
        for (Vector s : S) {
            double dist = Vectors.sqdist(x, s);
            if (dist > d) d = dist;
        }
        return d;
    }

    public static double exactMPD(ArrayList<Vector> S) {
        double dist = Double.NEGATIVE_INFINITY;
        for (Vector s : S) {
            for (Vector t : S) {
                double d = Vectors.sqdist(s, t);
                if (d > dist) dist = d;
            }
        }
        return dist;
    }

    public static double twoApproxMPD(ArrayList<Vector> S, int k) {
        long seed = 1218949;
        double dist = 0;

        ArrayList<Vector> T = new ArrayList<>();
        ArrayList<Vector> C;

        Random rand = new Random();
        rand.setSeed(seed);

        C = S;
        for (int i = 0; i < k; i++) {
            int l = rand.nextInt();
            T.add(C.get(l));
            C.remove(l);
        }

        for (Vector s : S) {
            for (Vector t : T) {
                double d = Vectors.sqdist(s, t);
                if (d > dist) dist = d;
            }
        }

        return dist;
    }

    // K-CENTER MPD
    private static ArrayList<Vector> kCenterMPD(ArrayList<Vector> PS, int k) {
        ArrayList<Vector> P;
        ArrayList<Vector> C = new ArrayList<>();

        P = PS;

        C.add(P.remove((int) (Math.random() * P.size())));             // choose the first center as a random point of P

        for (int i = 2; i <= k; i++) {                              // find the other k-1 centers
            double max = 0;
            Vector c = P.get(0);
            for (Vector p : P) {                                      // find the point c in P - C that maximizes d(c,C)
                double distance = distanceFromSet(p, C);
                if (distance > max) {
                    max = distance;
                    c = p;
                }
            }
            C.add(P.remove(P.indexOf(c)));                          /* Update C adding the new center c. To obtain P - C,
                                                                       each c added in C is removed from P */
        }
        return C;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Auxiliary methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    public static void main(String[] args) {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Reading points from a file whose name is provided as args[0]
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        String filename = args[0];
        ArrayList<Vector> inputPoints;

        try {

            inputPoints = readVectorsSeq(filename);

            int k = Integer.parseInt(args[1]);

            long start = System.currentTimeMillis();
            double exactMPD = exactMPD(inputPoints);
            long end = System.currentTimeMillis();

            System.out.println("EXACT ALGORITHM");
            System.out.println("Max distance = " + exactMPD);
            System.out.println("Running time = " + (end - start) + "ms");

            start = System.currentTimeMillis();
            double twoApproxMPD = twoApproxMPD(inputPoints, k);
            end = System.currentTimeMillis();

            System.out.println("2-APPROXIMATION ALGORITHM");
            System.out.println("k = " + k);
            System.out.println("Max distance = " + twoApproxMPD);
            System.out.println("Running time = " + (end - start) + "ms");

            start = System.currentTimeMillis();
            ArrayList<Vector> kCenterMPD = kCenterMPD(inputPoints, k);
            double exact_kCenterMPD = exactMPD(kCenterMPD);
            end = System.currentTimeMillis();
            System.out.println("k-CENTER-BASED ALGORITHM");
            System.out.println("k = " + k);
            System.out.println("Max distance = " + exact_kCenterMPD);
            System.out.println("Running time = " + (end - start) + "ms");

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}