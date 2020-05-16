import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.Arrays;
import java.util.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class G05HW2 {

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

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Developed methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    private static double distanceFromSet(Vector x, ArrayList<Vector> S) {
        double d = Double.NEGATIVE_INFINITY;                        // initialize to infinity
        for (Vector s : S) {
            double dist = Math.sqrt(Vectors.sqdist(x, s));          // compute the squared root of a point from a point set
            if (dist > d) d = dist;
        }
        return d;
    }

    public static double exactMPD(ArrayList<Vector> S) {            // first required algorithm
        double dist = Double.NEGATIVE_INFINITY;                     // initialize to infinity
        for (Vector s : S) {
            for (Vector t : S) {
                double d = Math.sqrt(Vectors.sqdist(s, t));
                if (d > dist) dist = d;
            }
        }
        return dist;
    }

    public static double twoApproxMPD(ArrayList<Vector> S, int k) {     // second required algorithm

        long seed = 1218949;
        double dist = 0;

        ArrayList<Vector> T = new ArrayList<>();    // set of k random points
        ArrayList<Vector> C;                        // clone of S, useful to avoid duplicate selected point

        Random rand = new Random();
        rand.setSeed(seed);

        C = S;
        for (int i = 0; i < k; i++) {
            int l = (int) Math.abs(rand.nextInt() % Math.sqrt(C.size()));       // select a random int

            Vector c = C.get(l);       // extract the point with the random index computed before
            T.add(c);                  // add the point to T
            C.remove(c);               // remove the point from C, to avoid selecting it again
        }

        for (Vector s : S) {
            for (Vector t : T) {
                double d = Math.sqrt(Vectors.sqdist(s, t));
                if (d > dist) dist = d;
            }
        }

        return dist;
    }

    private static ArrayList<Vector> kCenterMPD(ArrayList<Vector> PS, int k) {      // third required algorithm
        ArrayList<Vector> P;
        ArrayList<Vector> C = new ArrayList<>();

        P = PS;

        C.add(P.remove((int) (Math.random() * P.size())));          // choose the first center as a random point of P

        for (int i = 2; i <= k; i++) {                              // find the other k-1 centers
            double max = 0;
            Vector c = P.get(0);
            for (Vector p : P) {                                    // find the point c in P - C that maximizes d(c,C)
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

    // The following methods were developed to speed up the running
    // time of exactMPD method, since it takes hours to process the
    // medium and large datasets

    public static double cross(Point O, Point A, Point B) {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    }

    public static Point[] convexHull(Point[] P) {

        if (P.length > 1) {
            int n = P.length, k = 0;
            Point[] H = new Point[2 * n];

            Arrays.sort(P);

            // Build lower hull
            for (Point point : P) {
                while (k >= 2 && cross(H[k - 2], H[k - 1], point) <= 0.0)
                    k--;
                H[k++] = point;
            }

            // Build upper hull
            for (int i = n - 2, t = k + 1; i >= 0; i--) {
                while (k >= t && cross(H[k - 2], H[k - 1], P[i]) <= 0)
                    k--;
                H[k++] = P[i];
            }
            if (k > 1) {
                H = Arrays.copyOfRange(H, 0, k - 1); // remove non-hull vertices after k; remove k - 1 which is a duplicate
            }
            return H;
        } else if (P.length <= 1) {
            return P;
        } else {
            return null;
        }
    }

    public static double exactMPD_convexHull(Point[] P) {
        double dist = Double.NEGATIVE_INFINITY;
        for (Point x : P) {
            for (Point y : P) {
                double d = Math.sqrt(Math.pow(x.x - y.x, 2) + Math.pow(x.y - y.y, 2));
                if (d > dist) dist = d;
            }
        }

        return dist;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Code
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) {

        String filename = args[0];      // dataset name
        long start, end;                // helps with time statistics

        try {

            ArrayList<Vector> inputPoints = readVectorsSeq(filename);   // gathering of all dataset points
            int k = Integer.parseInt(args[1]);                          // set the value of k

            // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            // This commented part applies a conversion to a predefined class called Point. After that
            // we use a geometry property of Set of Points to compute the Convex Hull in the
            // exactMPD_for ConvexHull() method. Using this method prevent the explosion of the
            // required time for computing the max pairwise distance between points.
            // That is because the method runs in O(n*logn), while the exactMPD implemented from line 225
            // runs in O(n^2), where n is the input size of our Pointset.
            // Please feel free to remove the comment from line 193 to line 223
            // In our tests the exactMPD_convexHull took around 1000ms to complete and return the right distance
            // while the exactMPD took almost 10 hours (both for the uber-large dataset)
            // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

            //-----------------------------------------------------------------------------
            /*
            System.out.println("Start conversion from Vector to Point");
            start = System.currentTimeMillis();

            Point[] p = new Point[inputPoints.size()];
            for (int i = 0; i < inputPoints.size(); i++) {
                Vector v = inputPoints.get(i);
                double[] o = v.toArray();
                p[i] = new Point();
                p[i].x = o[0];
                p[i].y = o[1];
            }

            end = System.currentTimeMillis();

            System.out.println("\tInput size : " + p.length);
            System.out.println("\tConversion to Point finish in : " + (end - start) + "ms");

            System.out.println("Start finding Convex Hull");

            start = System.currentTimeMillis();
            Point[] hull = convexHull(p).clone();
            System.out.println("\tFinding Convex Hull diameter : ");
            System.out.println("\tNumber of vertices in Convex Hull : " + hull.length + "\n");
            double exactMPD_convexHull = exactMPD_convexHull(hull);
            end = System.currentTimeMillis();

            System.out.println("EXACT ALGORITHM (with Convex Hull)");
            System.out.println("Max distance = " + exactMPD_convexHull);
            System.out.println("Running time = " + (end - start) + "ms\n");
            */
            //-----------------------------------------------------------------------------
            System.out.println("EXACT ALGORITHM");
            start = System.currentTimeMillis();
            double exactMPD = exactMPD(inputPoints);
            end = System.currentTimeMillis();

            System.out.println("Max distance = " + exactMPD);
            System.out.println("Running time = " + (end - start) + "ms\n");
            //-----------------------------------------------------------------------------
            System.out.println("2-APPROXIMATION ALGORITHM");
            start = System.currentTimeMillis();
            double twoApproxMPD = twoApproxMPD(inputPoints, k);
            end = System.currentTimeMillis();

            System.out.println("k = " + k);
            System.out.println("Max distance = " + twoApproxMPD);
            System.out.println("Running time = " + (end - start) + "ms\n");
            //-----------------------------------------------------------------------------
            System.out.println("k-CENTER-BASED ALGORITHM");
            start = System.currentTimeMillis();
            ArrayList<Vector> kCenterMPD = kCenterMPD(inputPoints, k);
            double exact_kCenterMPD = exactMPD(kCenterMPD);
            end = System.currentTimeMillis();

            System.out.println("k = " + k);
            System.out.println("Max distance = " + exact_kCenterMPD);
            System.out.println("Running time = " + (end - start) + "ms\n");
            //-----------------------------------------------------------------------------

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}


class Point implements Comparable<Point> {

    double x, y;

    public int compareTo(Point p) {
        return Double.compare(this.x, p.x);
    }

    public String toString() {
        return "(" + x + "," + y + ")";
    }

}