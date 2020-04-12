import javafx.concurrent.Task;
import org.apache.log4j.lf5.LogLevel;
import org.apache.spark.SparkConf;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
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

        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> docs = sc.textFile(args[1]).repartition(K);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Long> count;
        JavaPairRDD<String, Long> tmp;

        Random randomGenerator = new Random();
        count = docs
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)

                    // Each row of the document is split by " "
                    String[] tokens = document.split(" ");
                    ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();

                    int index = 0;

                    // Scan all the token in each row
                    for (String token : tokens) {
                        // Add the token only if it's not a number
                        if (token.matches("[0-9 ]+")) {
                            index = Integer.parseInt(token);
                        } else {
                            pairs.add(new Tuple2<>((index % K), new Tuple2<>(token, 1L)));
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
        for (Tuple2<String, Long> o_pairs : count.sortByKey().collect()) {
            o.append(o_pairs).append(" ");
        }
        System.out.println("Output pairs: " + o);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // IMPROVED WORD COUNT with mapPartitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = docs
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

        // TODO understand how to use RDD function to get interesting info about data
        // Temporary Map used to clone the RDD entry and do transformation on them
        Map<String, Long> count_map = count.sortByKey().collectAsMap();
        Tuple2<String, Long> max_pair = computeMax(count_map);

//        tmp= count.filter(w -> w._1.contains("maxPartitionSize"));
//        System.out.println(tmp.max(Comparator.comparingLong(Tuple2::_2$mcC$sp))._2);
//        System.out.println("~"+count.subtractByKey(tmp).sortByKey().collect());
        Long n_max = 0L; // n_max store the highest class count
        for (int i = 0; i < K; i++) {
            for (Tuple2<String, Long> c : count.sortByKey().collect()) {
                if (c._1().equals("maxPartitionSize" + i)) {
                    if (c._2() >= n_max) n_max = c._2();
                }
            }

            // Removing the class "maxPartitionSize{id} to avoid possible
            // misleading information when finding the real max class count
            count_map.remove("maxPartitionSize" + i);

        }

        System.out.println("VERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent class = " + max_pair);
        System.out.println("Max partition size = " + n_max);

    }

    /**
     * <p>
     * computeMax compute all the tuples with max class count
     * and sort them in alphabetical order if there's tuples
     * that contains same max value.
     * Remember that [A-Z] < [a-z]
     * </p>
     *
     * @param map the mapped version of the RDD
     * @return the entry with max class count value and min lexicographic key
     */
    public static Tuple2<String, Long> computeMax(Map<String, Long> map) {

        // Support ArrayList that will contain all the classes with the same count
        ArrayList<Tuple2<String, Long>> m = new ArrayList<>();

        // max value contains max(class count inside the input map)
        Map.Entry<String, Long> max_value = Collections.max(map.entrySet(), Map.Entry.comparingByValue());

        // Foreach map entry, if there's more than one class with max_value, add it into the support ArrayList
        for (Map.Entry<String, Long> e : map.entrySet()) {
            if (max_value.getValue().equals(e.getValue())) {
                m.add(new Tuple2<>(e.getKey(), e.getValue()));
            }
        }

        // Sort the key added before in lexicographic order
        m.sort(Comparator.comparing(Tuple2::_1));

        // Return the first tuple
        return m.get(0);
    }

}