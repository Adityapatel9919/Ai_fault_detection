
"""predict.py - Prediction module"""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from .save_load import ModelArtifactManager
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
logger=logging.getLogger(__name__)
class FaultPredictor:
    def __init__(self,model_name='xgboost',model_store='results/model_store'):
        self.manager=ModelArtifactManager(model_store)
        v=self.manager.get_latest_version(model_name)
        self.model,self.metadata=self.manager.load_model(model_name,v,True)
        self.features=self.metadata.features
        self.targets=self.metadata.targets
    def _prepare(self,df):
        df=df.copy()
        for c in self.targets:
            if c in df.columns: df=df.drop(columns=c)
        miss=[c for c in self.features if c not in df.columns]
        if miss: raise ValueError(f'Missing features: {miss}')
        return df[self.features]
    def predict_dataframe(self,df):
        X=self._prepare(df)
        t=time.perf_counter()
        pred=self.model.predict(X)
        dt=(time.perf_counter()-t)*1000
        out=pd.DataFrame(pred,columns=['pred_sc_type','pred_fault_target','pred_phase_select'])
        if hasattr(self.model,'predict_proba'):
            try:
                probs=self.model.predict_proba(X)
                for i,n in enumerate(self.targets):
                    out[f'confidence_{n}']=np.max(probs[i],axis=1)
            except Exception:
                pass
        logger.info("Latency %.3f ms total",dt)
        return out
    def predict_file(self,input_path,output_path='predictions.csv'):
        p=Path(input_path)
        df=pd.read_parquet(p) if p.suffix.lower()=='.parquet' else pd.read_csv(p)
        out=self.predict_dataframe(df)
        out.to_csv(output_path,index=False)
        return out
    def predict_sample(self,sample:Dict[str,Any]):
        return self.predict_dataframe(pd.DataFrame([sample])).iloc[0].to_dict()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='xgboost')
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',default='predictions.csv')
    a=ap.parse_args()
    p=FaultPredictor(a.model)
    print(p.predict_file(a.input,a.output).head())
if __name__=='__main__':
    main()