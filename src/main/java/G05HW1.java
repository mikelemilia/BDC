import org.apache.spark.SparkConf;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

public class G05HW1 {

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: number_partitions, <path to file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("First BDC HW 2020");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> pairStrings = sc.textFile(args[1]).repartition(K);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Long> classCount;
        JavaPairRDD<String, Long> partitionCount;

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Version with deterministic partitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        classCount = pairStrings
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)

                    // Each row of the document is split by " "
                    String[] tokens = document.split(" ");
                    ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();

                    int i = 0;

                    // Scan all the token in each row
                    for (String token : tokens) {
                        // Add the token only if it's not a number
                        if (token.matches("[0-9 ]+")) {
                            i = Integer.parseInt(token);
                        } else {
                            pairs.add(new Tuple2<>((i % K), new Tuple2<>(token, 1L)));
                        }
                    }

                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R1)
                .flatMapToPair((triplet) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    for (Tuple2<String, Long> c : triplet._2()) {
                        counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()   // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });

        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        StringBuilder o = new StringBuilder();
        for (Tuple2<String, Long> o_pairs : classCount.sortByKey().collect()) {
            o.append(o_pairs).append(" ");
        }
        System.out.println("Output pairs: " + o);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Version with Spark partition
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&

        classCount = pairStrings
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)

                    // Each row of the document is split by " "
                    String[] tokens = document.split(" ");
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    // Scan all the token in each row
                    for (String token : tokens) {
                        // Add the token only if it's not a number
                        if (!token.matches("[0-9 ]+")) {
                            pairs.add(new Tuple2<>(token, 1L));
                        }
                    }
                    // Insert the pair ("maxPartitionSize", n_max)
                    // Note that n_max = 1L cause each tokens had only one word
                    pairs.add(new Tuple2<>("maxPartitionSize" + TaskContext.getPartitionId(), (long) pairs.size()));
                    return pairs.iterator();
                })
                .mapPartitionsToPair((cc) -> {    // <-- REDUCE PHASE (R1)

                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    while (cc.hasNext()) {
                        Tuple2<String, Long> tuple = cc.next();
                        counts.put(tuple._1(), tuple._2() + counts.getOrDefault(tuple._1(), 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }

                    return pairs.iterator();
                })
                .groupByKey() // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });

        partitionCount = classCount.filter(w -> w._1.contains("maxPartitionSize"));
        Tuple2<String, Long> n_max = partitionCount.max(new tupleComparator());
        Tuple2<String, Long> max_pair = classCount.subtractByKey(partitionCount).sortByKey().max(new tupleComparator());

        System.out.println("VERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent class = " + max_pair);
        System.out.println("Max partition size = " + n_max._2);

    }

    public static class tupleComparator implements Serializable, Comparator<Tuple2<String, Long>> {

        @Override
        public int compare(Tuple2<String, Long> tuple1, Tuple2<String, Long> tuple2) {

            if (tuple1._2 < tuple2._2) return -1;
            else if (tuple1._2 > tuple2._2) return 1;

            else {
                // When count is equal we return the smaller class in alphabetical order
                if (tuple2._1.compareTo(tuple1._1) < 0) return -1;
                else if (tuple2._1.compareTo(tuple1._1) > 0) return 1;
                return 0;
            }
        }

    }

}