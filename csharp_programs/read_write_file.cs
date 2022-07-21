using System;
using System.IO;
using System.Text;

public class ReadWriteFile {
    public static void Main(string[] args) {
        StreamWriter sw = new StreamWriter("../csharp_programs/test.txt", true, Encoding.UTF8);

        sw.WriteLine("Test!");

        sw.Close();
    }
}
