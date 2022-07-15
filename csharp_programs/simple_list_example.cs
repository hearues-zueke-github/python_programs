using System;
using System.Collections.Generic;

public delegate void PrintList<T>(in string name, in List<T> list, in bool with_new_line=true);

public class SimpleListExample {
    public static void Main(string[] args) {
        PrintList<int> print_list_int = delegate(in string name, in List<int> list, in bool with_new_line ) {
            Console.Write( name + ": " + string.Join( ", ", list.ToArray() ) );
            if (with_new_line) {
                Console.Write( "\n" );
            }
        };

        PrintList<long> print_list_long = delegate(in string name , in List<long> list, in bool with_new_line ) {
            Console.Write( name + ": " + string.Join( ", ", list.ToArray() ) );
            if (with_new_line) {
                Console.Write( "\n" );
            }
        };

        var list_int = new List<int>() { 1, 2, 3 };
        print_list_int("list_int", list_int);

        list_int.Add(6);
        print_list_int("list_int", list_int);

        var list_long = new List<long>() { 5, 7, 8, 6, 9 };
        print_list_long("list_long", list_long);

        int index_of_val = list_long.FindIndex( x => x == 6 );
        Console.WriteLine( "index_of_val: " + index_of_val );

        list_long.RemoveAt( index_of_val );
        print_list_long("list_long", list_long);
    }
}
