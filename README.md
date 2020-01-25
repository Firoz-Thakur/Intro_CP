# hellow-world
#code game

public class test55 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
        int x;
        printx();
        public void printx()
  	  {
  		  System.out.println(x);
  	  }
	}
}


//write a program to take input from the user to fill the elemennt in the array calculate the sum of element
//using enhance for loop.
import java.util.Scanner;
public class lop1 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
      int arr[]=new int[5];
      Scanner sc=new Scanner(System.in);
      for(int i=0;i<5;i++)
      {
     	arr[i]=sc.nextInt(); 
	  }
    for(int a:arr)
    	  System.out.println(a);
}
}



public class loop7 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int a; 
		a=7;
		while(true)
		 {   
			if(a>4)
		 {
			 break; 
		 }
			System.out.println("baba");
		 }
			 System.out.println("pu");	
	}

}



import java.util.*;

class conversion2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
       Demo obj=new Demo();
       obj.presentMenu();
	}
}
	
class Demo{
	
	 public static void presentMenu()
	 {
	  System.out.println("1.Create bank account");
	  System.out.println("2.Add money");
	  System.out.println("3.withdraw money");
	  System.out.println("4.exit");  
	  Scanner sc=new Scanner(System.in);
          int x=sc.nextInt();
	  switch(x)
	  {
	  case 1: 
		    System.out.println("account created");
                      break;
	  case 2: 
		    System.out.println("account created");
	              break;
	  case 3: 
		    System.out.println("account created");
	                break;
	  case 4: 
		    System.exit(0);
	  }
	 }
}




