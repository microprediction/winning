#include <stdio.h>       
#include <stdlib.h>
#include <string.h> /*for memcpy command*/
#include "movieobj.h"
#include "video.h"

#include <mem.h>
#include "exptlib.h"


#define status STAND_ALONE
#define numtrials 880
#define trialsperblock 40
#define numblocks ((numtrials)/(trialsperblock))
#define backup 0
#define fore_time 30
#define delta 11

void hline(int len, unsigned char far *b,long *seed);
void vhline(int len, int y, unsigned char far *b,long *seed);


int offset4[3]={0,4,8};
int offset8[2]={0,4};
/*int length[12]={23,39,57,77,99,123,148,174,201,230,260,291}; old*/
int length[12]={23,32,43,57,73,93,116,143,174,210,251,298};

int order[13]={2,5,0,9,3,6,1,8,10,4,7,11,12};
int xb[13][2]={{0,2},{1,3},{2,4},{3,5},{4,6},{5,7},{6,8},
		{7,9},{8,10},{9,11},{0,10},{1,11},{5,7}};

main()
{
  int *stim,*blktype;
  int b,s,t;

  logtype *log;
  FILE *outfile,*backfile;
  char *fontlist,junk[3];
  char backstr[40],outline[80];
  long seed;

  image *wrong_i,*blank, *targ_i;
  movie *wrong_m,*ready_m,*stim_m,*feed_m;
  image *ready12_i,*ready2_i;
  image *ready8a_i,*ready8b_i;
  response *data; int resp; long RT;

  fontlist=getenv("FONTLIST");
  log=(logtype *)malloc(1*sizeof(logtype));
  gen_init(status,log);
  SetupMoviePackage(fontlist);
  seed=log->seed;
  outfile=fopen(log->outfn,"wt");
  if (backup)
    {
     sprintf(backstr,"c:\\expbak\\%s",log->outfn);
     backfile=fopen(backstr,"wt");
    }

  blktype= (int *)malloc(sizeof(int)*numtrials);
  stim= (int *)malloc(sizeof(int)*numtrials);
  setstim(stim,blktype,&seed); 

 /* set up pallete here*/
  makePalette(GRAYSCALE);
  /*READY movie*/
  /*wrong movie*/
  wrong_i=newImage();
  downloadImage(wrong_i);
  drawText("invalid key",0,0,0,255);
  uploadImage(wrong_i);
  wrong_m=initMovie(1);
  setMovie(wrong_m,0,wrong_i,24);

  ready12_i=newImage();
  downloadImage(ready12_i);
  for (s=0;s<12;s++)
  {
    sprintf(junk,"%d",s+1);
    vhline(length[s],(s*16),picBuf,&seed);
    drawText(junk,xcenter,(s*16)+2,0,255);
  }
  uploadImage(ready12_i);
  ready2_i=newImage();


  ready_m=initMovie(1);



  stim_m=initMovie(2);
  feed_m=initMovie(1);
  blank=newImage();
  targ_i=newImage();
  setMovie(stim_m,0,blank,fore_time);



  for (b=0;b<numblocks;b++)
    {
      switch (blktype[b*(trialsperblock)]){
	  case 2: 
	    {
	      downloadImage(ready2_i);
	      clearPicBuf();
	      for (s=0;s<2;s++)
		 {
		   sprintf(junk,"%d",xb[order[b-6]][s]+1);
		   vhline(length[s],(xb[order[b-6]][s]*16),picBuf,&seed);
		   drawText(junk,xcenter,(xb[order[b-6]][s]*16)+2,0,255);
		  }
	       uploadImage(ready2_i);
	    }
	  setMovie(ready_m,0,ready2_i,1);break;
	  default : setMovie(ready_m,0,ready12_i,1);break;}
      runMovie(ready_m,UNTIL_RESPONSE,1);
      for (s=0;s<trialsperblock;s++)
	{
	  t=s+(b*(trialsperblock));
	  downloadImage(targ_i);
	  clearPicBuf();
	  hline(length[stim[t]],picBuf,&seed);
	  uploadImage(targ_i);
	  setMovie(stim_m,1,targ_i,1);
	  data=runMovie(stim_m,UNTIL_RESPONSE,1);
	  RT=data->x[0].rt-(msperframe*fore_time);
	  switch (data->x[0].resp){
	  case '1': resp=0;break;
	  case '2': resp=1;break;
	  case '3': resp=2;break;
	  case '4': resp=3;break;
	  case '5': resp=4;break;
	  case '6': resp=5;break;
	  case '7': resp=6;break;
	  case '8': resp=7;break;
	  case '9': resp=8;break;
	  case '0': resp=9;break;
	  case '-': resp=10;break;
	  case '=': resp=11;break;
	  case '@': resp=12;break;
	  default : resp=13;audio(INVALID,wrong_m);break;}
	  if (resp==stim[t]) audio(CORRECT);
	  else
	    {
	      sprintf(junk,"%d",(stim[t])+1);
	      downloadImage(targ_i);
	      drawText(junk,xcenter-10,ycenter+60,0,255);
	      uploadImage(targ_i);
	      setMovie(feed_m,0,targ_i,30);
	      runMovie(feed_m,FULL_SHOW,0);
	    }
	  sprintf(outline,"sub%03d  blk%02d  trl%03d  bt%1d %02d %02d %05d\n",
		  log->subjnum,b,s,blktype[t],stim[t],resp,RT);
	  fprintf(outfile, "%s",outline);
	  if (backup) fprintf(backfile,"%s",outline);
	  if (resp==12) {fclose(outfile);CleanupMoviePackage();
	  printf ("stopped while running by participant\n");exit(1);}
	}
    }
  fclose(outfile);
  if (backup) fclose(backfile);
  thankyou();
  CleanupMoviePackage();
}


	  





  /*-----------------------------------------*/










void vhline(int len, int y, unsigned char far *b,long *seed)
{
  int q, yrnd,xstart;
  char junk[3];
  
  yrnd=y;
  xstart=xcenter-(len/2);
  for (q=xstart;q<xstart+len;q++) b[o(q,yrnd)]=255;
}



void hline(int len, unsigned char far *b,long *seed)
{
  int q, yrnd,xstart;
  char junk[3];
  
  yrnd=ycenter+randint(-delta,delta,seed);
  xstart=xcenter-(len/2)+randint(-delta,delta,seed);
  for (q=xstart;q<xstart+len;q++) b[o(q,yrnd)]=255;
}


setstim(int *stim,int *blktype, long *seed)
{
  

  int c,b,t;
  int temp[120];
  long critseed;
  
  c=0;
  /*warmup*/
  for (t=0;t<120;t++)
    temp[t]=t%12;
  distribute(temp,120,seed);
  for (t=0;t<120;t++)
    {
      stim[c]=temp[t];
      blktype[c]=0;
      c++;
    }
  /*pretest*/
  critseed=*seed;
  for (t=0;t<120;t++)
    temp[t]=t%12;
  distribute(temp,120,seed);
  for (t=0;t<120;t++)
    {
      stim[c]=temp[t];
      blktype[c]=1;
      c++;
    }
  for (b=0;b<13;b++)
    {
     for (t=0;t<40;t++)
       temp[t]=t%2;
     distribute(temp,40,seed);
     for (t=0;t<40;t++)
      {
       stim[c]=xb[order[b]][temp[t]];
       blktype[c]=2;
       c++;
      }
     }
	     
  
  /*posttest*/
  *seed=critseed;
  for (t=0;t<120;t++)
    temp[t]=t%12;
  distribute(temp,120,seed);
  for (t=0;t<120;t++)
    {
      stim[c]=temp[t];
      blktype[c]=7;
      c++;
    }

  /*for (t=0;t<numtrials;t++)
    printf("%d %d %d\n",t,blktype[t],stim[t]);*/
}
 


