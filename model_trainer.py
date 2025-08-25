from datasets import load_dataset, concatenate_datasets
from dataset import load
from datasets import Dataset, Features, ClassLabel, Value
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import numpy.ma as ma
from transformers.data.data_collator import DataCollatorForTokenClassification
from tqdm import tqdm
import datetime
import random
from tools import print_cm
import torch

class ModelTrainer():
    def __init__(self, task:int, model:str,run_name:str, data_percentage:float,use_token_type_ids:bool, opimizer_config, tokenizer_config,languages,do_hyperparameter_search = False, **args):
        self.task = task 
        self.model_checkpoint = model
        self.run_name = run_name
        self.batch_size = 8
        self.label_all_tokens = True
        self.data_factor = data_percentage # train and test on x percent of the data
        self.opimizer_config = opimizer_config
        self.tokenizer_config = tokenizer_config
        self.languages = languages
        self.use_token_type_ids = use_token_type_ids
        self.do_hyperparameter_search = do_hyperparameter_search
        if self.task == 1:    
            self.label_2_id = {"0":0, "1":1}
        else:
            self.label_2_id = {"0":0, ".":1, ",":2, "?":3, "-":4, ":":5} 
            
        self.id_2_label = list(self.label_2_id.keys())        
    
    def tokenize_and_align_data(self,data,stride=0):
        if self.model_checkpoint == "camembert/camembert-large":
            # this model has a wrong maxlength value, so we need to set it manually
            self.tokenizer.model_max_length = 512
            
        tokenizer_settings = {'is_split_into_words':True,'return_offsets_mapping':True, 
                                'padding':False, 'truncation':True, 'stride':stride, 
                                'max_length':self.tokenizer.model_max_length, 'return_overflowing_tokens':True}
        tokenized_inputs = self.tokenizer(data[0], **tokenizer_settings)

        labels = []
        for i,document in enumerate(tokenized_inputs.encodings):
            doc_encoded_labels = []
            last_word_id = None
            for word_id  in document.word_ids:            
                if word_id == None: #or last_word_id == word_id:
                    doc_encoded_labels.append(-100)        
                else:
                    #document_id = tokenized_inputs.overflow_to_sample_mapping[i]
                    #label = examples[task][document_id][word_id]
                    label = data[1][word_id]
                    doc_encoded_labels.append(self.label_2_id[label])
                last_word_id = word_id
            labels.append(doc_encoded_labels)
        
        tokenized_inputs["labels"] = labels  
        # print("tokenized_inputs: ", tokenized_inputs)  
        # print("Tokenize and align data: Completed")
        return tokenized_inputs

    def to_dataset(self,data,stride=0):
        print("to_dataset: Entered")
        # Print data example
        print("data: ", data)

        labels, token_type_ids, input_ids, attention_masks = [],[],[],[]
        for item in tqdm(data):
            result = self.tokenize_and_align_data(item,stride=stride)        
            labels += result['labels']
            if self.use_token_type_ids:
                token_type_ids += result['token_type_ids']
            input_ids += result['input_ids']
            attention_masks += result['attention_mask']

            # print("labels: ", result['labels'])
            # print("input ids: ", result['input_ids'])
            # print("attention masks: ", result['attention_mask'])

            # input("Press Enter to continue...")  # Debugging pause

        if self.use_token_type_ids:
            return Dataset.from_dict({'labels': labels, 'token_type_ids':token_type_ids, 'input_ids':input_ids, 'attention_mask':attention_masks})
        else:
            return Dataset.from_dict({'labels': labels, 'input_ids':input_ids, 'attention_mask':attention_masks})

    def compute_metrics_generator(self):    
        def metrics(pred):
            mask = np.less(pred.label_ids,0)    # mask out -100 values
            labels = ma.masked_array(pred.label_ids,mask).compressed() 
            preds = ma.masked_array(pred.predictions.argmax(-1),mask).compressed() 
            if self.task == 1:
                precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")  
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")  
                print("\n----- report -----\n")
                report = classification_report(labels, preds,target_names=self.label_2_id.keys())
                print(report)
                print("\n----- confusion matrix -----\n")
                cm = confusion_matrix(labels,preds,normalize="true")
                print_cm(cm,self.id_2_label)

            acc = accuracy_score(labels, preds)    
            return {     
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy':acc,        
            }
        return metrics   

    def run_training(self):
        val_data = []
        train_data = []

        for language in self.languages:
            val_data += load("data/sepp_nlg_2021_train_dev_data_v5.zip","dev",language,subtask=self.task)
            train_data += load("data/sepp_nlg_2021_train_dev_data_v5.zip","train",language,subtask=self.task)

        #todo: implement augmentaion        
        aug_data =[]# load("data/bundestag_aug.zip","aug","de",subtask=task)
        #aug_data += load("data/leipzig_aug_de.zip","aug","de",subtask=task)
        ## tokenize data
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint,**self.tokenizer_config)

        #train_data = train_data[:int(len(train_data)*data_factor)] # limit data to x%
        #aug_data = aug_data[:int(len(aug_data)*data_factor)] # limit data to x%
        print("tokenize training data")
        tokenized_dataset_train = self.to_dataset(train_data,stride=100)

        del train_data
        #tokenized_dataset_aug = to_dataset(aug_data,stride=100)
        #del aug_data
        if self.data_factor < 1.0:
            train_split = tokenized_dataset_train.train_test_split(train_size=self.data_factor)
            tokenized_dataset_train = train_split["train"]
            #aug_split = tokenized_dataset_aug.train_test_split(train_size=data_factor)
            #tokenized_dataset_aug = aug_split["train"]

        #tokenized_dataset_train = concatenate_datasets([tokenized_dataset_aug,tokenized_dataset_train])
        tokenized_dataset_train.shuffle(seed=42)

        print("tokenize validation data")
        val_data = val_data[:int(len(val_data)*self.data_factor)] # limit data to x%
        tokenized_dataset_val = self.to_dataset(val_data)

        del val_data

        # Check mpc availability using torch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        print("Training model with run name: ", self.run_name)
        args = TrainingArguments(
            output_dir=f"models/{self.run_name}/checkpoints",
            run_name=self.run_name,    
            eval_strategy = "epoch",
            learning_rate=4e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=self.opimizer_config["num_train_epochs"],
            # adafactor=self.opimizer_config["adafactor"], 
            optim="adafactor", # use adafactor or adamw_torch
            #weight_decay=0.005,
            #weight_decay=2.4793153505992856e-11,
            #adam_epsilon=5.005649261324263e-10,
            warmup_steps=50,
            dataloader_pin_memory=False,
            #lr_scheduler_type="cosine",
            report_to=["tensorboard"],
            logging_dir='runs/'+self.run_name,            # directory for storing logs
            logging_first_step=True,
            logging_steps=100,
            save_steps=10000,
            save_total_limit=10,
            seed=16, 
            fp16=False,  # set to True if you have a GPU with FP16 support   
        )
        print("Training arguments created")
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        print("Data collator created")
        def model_init():
            return AutoModelForTokenClassification.from_pretrained(self.model_checkpoint, num_labels=len(self.label_2_id))

        print("Model init function created")
        # print("Training dataset features:")
        # print(tokenized_dataset_train.features)

        print("\nValidation dataset features:")
        # print(tokenized_dataset_val.features)

        # self.diagnose_trainer_issues(
        #     model_init=model_init,
        #     training_args=args,
        #     train_dataset=tokenized_dataset_train,
        #     eval_dataset=tokenized_dataset_val,
        #     data_collator=data_collator,
        #     compute_metrics=self.compute_metrics_generator())

        trainer = Trainer(
            model_init=model_init,
            args = args,    
            train_dataset=tokenized_dataset_train,
            eval_dataset=tokenized_dataset_val,
            data_collator=data_collator,
            # tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_generator()
        )

        if self.do_hyperparameter_search:
            print("----------hyper param search------------")
            return self.run_hyperparameter_search(trainer)
        else:
            trainer.train()
            trainer.save_model(f"models/{self.run_name}/final")
            return trainer.state.log_history

    def run_hyperparameter_search(self, trainer):
        import gc
        import torch
        def my_hp_space(trial):    
            gc.collect()
            torch.cuda.empty_cache()
            return {        
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 1,5),
                "seed": trial.suggest_int("seed", 1, 40),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8]),
                "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
                "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
                "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1,2,4,8]),
            }
        def my_objective(metrics):            
            return metrics['eval_f1']

        result = trainer.hyperparameter_search(direction="maximize",n_trials=200,hp_space=my_hp_space, compute_objective=my_objective)
        
        # print(result)
        return result
    
    from typing import Callable, Dict, Any, Optional
    from transformers.modeling_utils import PreTrainedModel
    from transformers.trainer import Trainer
    
    def diagnose_trainer_issues(
        self,
        model_init: Callable[[], PreTrainedModel],
        training_args: 'TrainingArguments',
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None
    ) -> None:
        print("--- Running Trainer Component Diagnostic ---")
        all_checks_passed = True

        try:
            # --- Check 1: Dataset Columns ---
            print("\n[1/5] Checking Dataset Columns...")
            required_cols = {'input_ids', 'attention_mask', 'labels'}
            train_cols = set(train_dataset.column_names)
            if not required_cols.issubset(train_cols):
                all_checks_passed = False
                print(f"  [FAIL] Training dataset is missing required columns!")
                print(f"         - Missing: {required_cols - train_cols}")
                print(f"         - Available: {train_cols}")
            else:
                print("  [OK] Training dataset has all required columns.")

            if eval_dataset:
                eval_cols = set(eval_dataset.column_names)
                if not required_cols.issubset(eval_cols):
                    all_checks_passed = False
                    print(f"  [FAIL] Evaluation dataset is missing required columns!")
                    print(f"         - Missing: {required_cols - eval_cols}")
                    print(f"         - Available: {eval_cols}")
                else:
                    print("  [OK] Evaluation dataset has all required columns.")

            # --- Check 2: Dataset Content ---
            print("\n[2/5] Checking Dataset Content...")
            if len(train_dataset) == 0:
                all_checks_passed = False
                print("  [FAIL] The training dataset is empty.")
            else:
                print(f"  [OK] Training dataset contains {len(train_dataset)} rows.")
            if eval_dataset:
                if len(eval_dataset) == 0:
                    print("  [WARN] The evaluation dataset is empty.")
                else:
                    print(f"  [OK] Evaluation dataset contains {len(eval_dataset)} rows.")

            # --- Check 3: Label Configuration vs. Data ---
            print("\n[3/5] Checking Model Labels vs. Dataset Labels...")
            temp_model = model_init()
            model_num_labels = temp_model.config.num_labels

            unique_labels_train = set()
            for example in train_dataset:
                unique_labels_train.update(example['labels'])
            
            all_unique_labels = unique_labels_train
            if eval_dataset:
                unique_labels_val = set()
                for example in eval_dataset:
                    unique_labels_val.update(example['labels'])
                all_unique_labels = unique_labels_train.union(unique_labels_val)

            if not all_unique_labels:
                print("  [FAIL] No labels found in datasets. The 'labels' columns might be empty.")
                all_checks_passed = False
            else:
                max_label_in_data = max(all_unique_labels)
                print(f"  - Model is configured with num_labels = {model_num_labels}")
                print(f"  - Maximum label ID found in data is {max_label_in_data}")
                if model_num_labels <= max_label_in_data:
                    all_checks_passed = False
                    print("  [FAIL] Label mismatch found!")
                    print(f"         The model expects label indices from 0 to {model_num_labels - 1}, but your data contains label {max_label_in_data}.")
                    print(f"         SOLUTION: Update `num_labels` in your model config to be at least {max_label_in_data + 1}.")
                else:
                    print("  [OK] Model's `num_labels` is compatible with dataset labels.")

            # --- Check 4: FP16 Hardware Compatibility ---
            print("\n[4/5] Checking Hardware for FP16...")
            import torch
            if training_args.fp16:
                if not torch.cuda.is_available():
                    all_checks_passed = False
                    print("  [FAIL] `fp16=True` requires an NVIDIA GPU, but CUDA is not available.")
                    print("         SOLUTION: Set `fp16=False` in TrainingArguments or install a CUDA-enabled PyTorch.")
                else:
                    print("  [OK] CUDA is available for FP16 training.")
            else:
                print("  [INFO] FP16 training is not enabled.")
                
            # --- Check 5: Other Components ---
            print("\n[5/5] Checking Other Components...")
            if data_collator is None:
                print("  [WARN] `data_collator` is not provided. Trainer will use `default_data_collator`.")
            else:
                print("  [OK] `data_collator` is provided.")
                
            if compute_metrics is None:
                print("  [INFO] `compute_metrics` is not provided. No evaluation metrics will be computed beyond loss.")
            else:
                print("  [OK] `compute_metrics` is provided.")

            print("\n--- Diagnostic Complete ---")
            if all_checks_passed:
                print("✅ All checks passed. You should be able to initialize the Trainer without a `ValueError`.\n")
            else:
                print("❌ One or more critical checks failed. Please review the output above and fix the issues before initializing the Trainer.\n")

        except Exception as e:
            print(f"\n🚨 An error occurred during the diagnostic itself: {e}")
            print("   This might indicate a deeper problem with one of your components (e.g., `model_init` is crashing).")


if __name__ =="__main__":
    trainer = ModelTrainer(task=2,model="dbmdz/bert-base-italian-xxl-uncased",run_name="optim",data_percentage=0.1,use_token_type_ids=True, opimizer_config={"adafactor": False,"num_train_epochs": 3},tokenizer_config={"strip_accent": True, "add_prefix_space":False},languages=["it"], do_hyperparameter_search=True)
    result = trainer.run_training()

    print(result)