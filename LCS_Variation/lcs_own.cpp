#include <iostream>
#include <bits/stdc++.h>

using namespace std;

int dp[101][101];

int lcs(string s1,string s2,int n,int m)
{
    if(n==0 || m==0)
    {
        return 0;
    }
  if(dp[n][m]!=-1)
   return dp[n][m];
    
  if(s1[n-1]==s2[m-1])
  {
      dp[n][m]=1+ lcs(s1,s2,n-1,m-1);
  }
  else
  {
      dp[n][m]=max(lcs(s1,s2,n-1,m),lcs(s1,s2,n,m-1));
  }
     return dp[n][m]; 
    
}





int lcs(string s1,string s2,int n,int m)
{
  
  int dp[n+1][m+1];
  
  for(int i=0;i<=n;i++)
  {
      dp[i][0]=0;
  }
   
  for(int i=0;i<=m;i++)
  {
      dp[0][i]=0;
  }
  
  for(int i=1;i<=n;i++)
  {
      for(int j=1;j<=m;j++)
      {
          if(s1[i-1]==s2[j-1])
          {
              dp[i][j]=1+dp[i-1][j-1];
          }
          else
          {
              dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
          }
          
      }
  }
  
  
return dp[n][m];  
  
}

int main() {
	//code
	int t;
	cin>>t;
	while(t--)
	{
	    int n1,n2;
	    cin>>n1>>n2;
	    cin.get();
	    string s1,s2;
	    cin>>s1>>s2;
	    
	   // memset(dp,-1,sizeof(dp));
	   
	   int l=lcs(s1,s2,s1.size(),s2.size());
	   cout<<l<<endl;
	    
	}
	
	
	
	
	return 0;
}
