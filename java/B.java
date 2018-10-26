package myprograms;

import myprograms.A;

class B {  
  
  public static void main(String args[]) {  
 
    A a = new A();

    System.out.println("Hello World!");
    System.out.println("The line number is " + new Exception().getStackTrace()[0].getLineNumber());

  }  

}
