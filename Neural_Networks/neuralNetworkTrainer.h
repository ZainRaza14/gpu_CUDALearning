
#ifndef nntrain

#define nnTrain


#include <fstream>

#include <vector>

#include "nNet.h"

#define lR 0.001

#define m_epchs 1500

#define accur_d 90  




class nnTrain
{

private:

	
	nNet* NN;
	
	float l_R;	
	
	long eP;
	
	long max_eP;
	
	float d_acc;
	
	float** d_Inp;
	
	float** d_Out;

	float* h_errG;
	
	float* o_errG;

    float *d_Out1;
	
	float *inP;
    
    float *m_hidden;
	
	float *weights_2;
    
    float *o_errG1; 

	float train_Acc;
	
	float val_Acc;
	
	float gen_Acc;

	bool u_B;

	bool l_E;
	
	std::fstream logFile;
	
	int logR;
	
	int lastLog;

public:	
	
	nnTrain( nNet* untrainedNetwork );
	
	~nnTrain();

	
	void setTrain( double learningRate, bool m_batch );
	
	void setStop( int mEP, double dAcc);
	
	void useBatchLearning( bool flag ){ u_B = flag; }
	
	void e_Log( const char* filename, int resolution );

	void netTrain( trainingDataSet* trainSet );


private:
	
	inline float get_oerrG( float dVal, float oVal );
	
	float get_herrG( int j );
	
	float get_herrGB( int j, int b );
	
	void r_TrainEP( std::vector<dataEntry*> trainingSet , int eP);
	
	void backp_B(std::vector<float*> dOVec);
	
	void backp(float* desiredOutputs);
	
	void w_Update();
};


#endif
