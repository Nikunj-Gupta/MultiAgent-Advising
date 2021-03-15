all: 
	clear 
	python3 teacher.py --expname teacher  
	python3 student.py --expname student--0.01 --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.01 
	python3 student.py --expname student--0.02 --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.02 
	python3 student.py --expname student--0.03 --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.03  
	python3 student.py --expname student--0.04 --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.04 
	python3 student.py --expname student--0.05 --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.05  

new: 
	clear 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.20 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.19 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.18 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.17 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.16 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.15 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.14 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.13 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.12 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.11 
	python3 student.py --expname student --load_teacher models/save-dir/10000_teacher --confidence_threshold 0.10 


ADVICE = 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 #0.4 0.5 0.6 0.7 0.8 0.9 1.0 # 0.1 0.2 0.3 
student: 
	clear 
	$(foreach var,$(ADVICE),python3 student-dqn.py --expname newstudent --use_teacher 1 --advice_threshold $(var);) 

teacher: 
	clear 
	python3 student-dqn.py --use_teacher 0 --expname teacher 

evaluate: 
	clear 
	python3 evaluate.py 