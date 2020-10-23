(defn containsDuplicates [a] (= (count (into #{} a)) (count a)))
(def a [1 2 3 4 4 2])
(println (format "a: %s" a))

(def b (containsDuplicates a))
(println (format "b: %s" b))

(println (format "(count a): %s" (count a)))
