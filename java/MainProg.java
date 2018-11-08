package myprograms;

import myprograms.A;

class MainProg {  
  
  public static void main(String args[]) {  
 
    System.out.println("Hello World!");
    System.out.println("The line number is " + new Exception().getStackTrace()[0].getLineNumber());

  }  

}
