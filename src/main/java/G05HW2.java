import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

class Point implements Comparable<Point> {
    double x, y;

    public int compareTo(Point p) {
        if (this.x == p.x) {
            return 0;
        } else if(this.x<p.x){
            return -1;
        }
        else return 1;
    }

    public String toString() {
        return "(" + x + "," + y + ")";
    }

}

public class G05HW2 {

    private static double distanceFromSet(Vector x, ArrayList<Vector> S) {
        double d = Double.NEGATIVE_INFINITY;                        // initialize to infinity
        for (Vector s : S) {
            double dist = Math.sqrt(Vectors.sqdist(x, s));
            if (dist > d) d = dist;
        }
        return d;
    }

    public static double exactMPD(ArrayList<Vector> S) {
        double dist = Double.NEGATIVE_INFINITY;
        for (Vector s : S) {
            for (Vector t : S) {
                double d = Math.sqrt(Vectors.sqdist(s, t));
                if (d > dist) dist = d;
            }
        }
        return dist;
    }

    public static double cross(Point O, Point A, Point B) {
        return (A.x - O.x) *(B.y - O.y) - (A.y - O.y) *(B.x - O.x);
    }

    public static Point[] convex_hull(Point[] P) {

        if (P.length > 1) {
            int n = P.length, k = 0;
            Point[] H = new Point[2 * n];

            Arrays.sort(P);

            // Build lower hull
            for (int i = 0; i < n; ++i) {
                while (k >= 2 && cross(H[k - 2], H[k - 1], P[i]) <= 0.0)
                    k--;
                H[k++] = P[i];
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

    public static double exactMPD_forConvexHull(Point[] P)
    {
        double dist = Double.NEGATIVE_INFINITY;
        for(Point x : P)
        {
            for(Point y : P)
            {
                double d = Math.sqrt(Math.pow(x.x-y.x,2)+Math.pow(x.y-y.y,2));
                if(d>dist) dist=d;
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
            int l = (int) Math.abs(rand.nextInt() % Math.sqrt(C.size()));

            Vector c = C.get(l);
            T.add(c);
            C.remove(c);
        }

        for (Vector s : S) {
            for (Vector t : T) {
                double d = Math.sqrt(Vectors.sqdist(s, t));
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

            int n = inputPoints.size();

            System.out.println("Start Point conversion");
            long start = System.currentTimeMillis();
            Point[] p = new Point[n];
            for(int i = 0; i<n ; i++ )
            {
                Vector v = inputPoints.get(i);
                double[] o = v.toArray();
                p[i]= new Point();
                p[i].x = o[0];
                p[i].y = o[1];
            }

            long end = System.currentTimeMillis();
            System.out.println("Input Size= " + p.length);
            System.out.println("conversion to Point finish in= " + (end - start) + "ms" );

            System.out.println("Start finding Convex Hull");

            start = System.currentTimeMillis();
            Point[] hull = convex_hull(p).clone();
            System.out.println("Finding Convex Hull diameter.");
            System.out.println("Number of vertices in Convex Hull= " + hull.length);
            double exactMPD_forConvexHull = exactMPD_forConvexHull(hull);
            end = System.currentTimeMillis();

            System.out.println("EXACT ALGORITHM");
            System.out.println("Max distance = " + exactMPD_forConvexHull);
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