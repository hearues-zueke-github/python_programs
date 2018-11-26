package myprograms;

import myprograms.D;

class C {  
  
  public static void main(String args[]) {  
 
    D d = new D();

    System.out.println("Hello World!");
    System.out.println("The line number is " + new Exception().getStackTrace()[0].getLineNumber());

  }  

}
