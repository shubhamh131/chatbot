import json
import pickle
import numpy as np
import nltk
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ChatbotTrainer:
    """Enhanced Chatbot Training Class with improved architecture and monitoring"""
    
    def __init__(self, intents_file="intents.json", log_level=logging.INFO):
        self.intents_file = intents_file
        self.lemmatizer = WordNetLemmatizer()
        self.setup_directories()  # Create directories first
        self.setup_logging(log_level)  # Then setup logging
        self.download_nltk_data()
        
        # Training parameters
        self.config = {
            'epochs': 300,
            'batch_size': 16,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'l2_reg': 0.01,
            'validation_split': 0.2,
            'patience': 25,
            'min_lr': 1e-6
        }
        
    def setup_logging(self, level):
        """Setup enhanced logging with timestamps"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"training_{timestamp}.log"
        
        # Ensure logs directory exists (double safety check)
        Path("logs").mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(f"logs/{log_filename}"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 ChatbotTrainer initialized")
        self.logger.info("📁 Directories setup complete")
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['models', 'data', 'logs', 'plots', 'checkpoints']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        # Don't use logger here since it's not created yet
        print("📁 Directories setup complete")
        
    def download_nltk_data(self):
        """Download required NLTK data with progress tracking"""
        nltk_data = [
            ('punkt', 'Punkt tokenizer'),
            ('wordnet', 'WordNet lemmatizer'),
            ('omw-1.4', 'Open Multilingual Wordnet'),
            ('stopwords', 'Stopwords corpus')
        ]
        
        for data_name, description in nltk_data:
            try:
                nltk.download(data_name, quiet=True)
                self.logger.info(f"✅ Downloaded {description}")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to download {description}: {e}")
                
    def load_intents(self):
        """Load and validate intents with enhanced error handling"""
        try:
            if not os.path.exists(self.intents_file):
                self.logger.error(f"❌ Intents file not found: {self.intents_file}")
                return {"intents": []}
                
            with open(self.intents_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            if "intents" not in data:
                self.logger.error("❌ Invalid intents format: 'intents' key missing")
                return {"intents": []}
                
            # Enhanced validation
            valid_intents = []
            total_patterns = 0
            
            for i, intent in enumerate(data["intents"]):
                if not all(key in intent for key in ["tag", "patterns", "responses"]):
                    self.logger.warning(f"⚠️  Skipping invalid intent at index {i}")
                    continue
                    
                if not intent["patterns"] or not intent["responses"]:
                    self.logger.warning(f"⚠️  Empty patterns/responses in intent: {intent['tag']}")
                    continue
                    
                valid_intents.append(intent)
                total_patterns += len(intent["patterns"])
                
            data["intents"] = valid_intents
            self.logger.info(f"✅ Loaded {len(valid_intents)} intents with {total_patterns} patterns")
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON decode error: {e}")
            return {"intents": []}
        except Exception as e:
            self.logger.error(f"❌ Error loading intents: {e}")
            return {"intents": []}
            
    def preprocess_data(self, intents):
        """Enhanced data preprocessing with better text cleaning"""
        words = []
        classes = []
        documents = []
        ignore_chars = ['?', '!', '.', ',', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}']
        
        self.logger.info("🔄 Starting data preprocessing...")
        
        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                # Clean and tokenize
                pattern = pattern.lower().strip()
                if not pattern:
                    continue
                    
                word_list = nltk.word_tokenize(pattern)
                word_list = [w for w in word_list if w.isalpha() and len(w) > 1]
                
                if not word_list:
                    continue
                    
                words.extend(word_list)
                documents.append((word_list, intent["tag"]))
                
                if intent["tag"] not in classes:
                    classes.append(intent["tag"])
        
        # Enhanced word processing
        words = [self.lemmatizer.lemmatize(w.lower()) for w in words 
                if w not in ignore_chars and len(w) > 1]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        
        # Data quality check
        if len(words) < 50:
            self.logger.warning("⚠️  Small vocabulary size - consider adding more training data")
        if len(classes) < 3:
            self.logger.warning("⚠️  Few intent classes - model may not generalize well")
            
        self.logger.info(f"✅ Preprocessing complete: {len(words)} words, {len(classes)} classes")
        return words, classes, documents
        
    def create_training_data(self, words, classes, documents):
        """Create optimized training data with stratified sampling"""
        training = []
        output_empty = [0] * len(classes)
        
        self.logger.info("🔄 Creating training data...")
        
        # Track class distribution
        class_counts = {}
        
        for doc in documents:
            # Create bag of words
            bag = [0] * len(words)
            word_patterns = [self.lemmatizer.lemmatize(w.lower()) for w in doc[0]]
            
            for i, w in enumerate(words):
                if w in word_patterns:
                    bag[i] = 1
                    
            # Create output row
            output_row = list(output_empty)
            class_idx = classes.index(doc[1])
            output_row[class_idx] = 1
            
            training.append([bag, output_row])
            
            # Track class distribution
            class_counts[doc[1]] = class_counts.get(doc[1], 0) + 1
        
        # Log class distribution
        self.logger.info("📊 Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            self.logger.info(f"   {class_name}: {count} samples")
            
        # Shuffle and convert to numpy
        training = shuffle(training, random_state=42)
        train_x = np.array([item[0] for item in training], dtype=np.float32)
        train_y = np.array([item[1] for item in training], dtype=np.float32)
        
        self.logger.info(f"✅ Training data created: {train_x.shape}")
        return train_x, train_y
        
    def build_model(self, input_shape, output_shape):
        """Build enhanced neural network with regularization"""
        self.logger.info("🏗️  Building neural network model...")
        
        model = Sequential([
            # First hidden layer
            Dense(512, input_shape=(input_shape,), activation='relu',
                  kernel_regularizer=l2(self.config['l2_reg'])),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            # Second hidden layer
            Dense(256, activation='relu',
                  kernel_regularizer=l2(self.config['l2_reg'])),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            # Third hidden layer
            Dense(128, activation='relu',
                  kernel_regularizer=l2(self.config['l2_reg'])),
            BatchNormalization(),
            Dropout(self.config['dropout_rate'] / 2),
            
            # Output layer
            Dense(output_shape, activation='softmax')
        ])
        
        # Compile with optimized settings
        optimizer = Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Log model architecture
        self.logger.info("🏗️  Model architecture:")
        model.summary(print_fn=lambda x: self.logger.info(f"   {x}"))
        
        return model
        
    def setup_callbacks(self):
        """Setup training callbacks for monitoring and optimization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=self.config['min_lr'],
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                filepath=f'checkpoints/best_model_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            TensorBoard(
                log_dir=f'logs/tensorboard_{timestamp}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        return callbacks
        
    def train_model(self, model, train_x, train_y):
        """Enhanced training with stratified validation split"""
        self.logger.info("🚀 Starting model training...")
        
        # Stratified split to maintain class distribution
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.config['validation_split'], 
                                   random_state=42)
        train_idx, val_idx = next(sss.split(train_x, train_y.argmax(axis=1)))
        
        x_train, x_val = train_x[train_idx], train_x[val_idx]
        y_train, y_val = train_y[train_idx], train_y[val_idx]
        
        self.logger.info(f"📊 Training set: {x_train.shape[0]} samples")
        self.logger.info(f"📊 Validation set: {x_val.shape[0]} samples")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train the model
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=1,
            callbacks=callbacks,
            shuffle=True
        )
        
        return model, history, (x_val, y_val)
        
    def evaluate_model(self, model, x_val, y_val, classes):
        """Comprehensive model evaluation with metrics"""
        self.logger.info("📊 Evaluating model performance...")
        
        # Basic evaluation
        scores = model.evaluate(x_val, y_val, verbose=0)
        self.logger.info(f"✅ Validation Loss: {scores[0]:.4f}")
        self.logger.info(f"✅ Validation Accuracy: {scores[1]*100:.2f}%")
        self.logger.info(f"✅ Top-K Accuracy: {scores[2]*100:.2f}%")
        
        # Detailed classification report
        y_pred = model.predict(x_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        # Classification report
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=classes, output_dict=True)
        
        self.logger.info("📋 Classification Report:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                self.logger.info(f"   {class_name:15} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | Support: {support}")
        
        return scores, report
        
    def plot_training_history(self, history):
        """Enhanced training visualization"""
        self.logger.info("📈 Creating training visualizations...")
        
        # Fix for seaborn style - use compatible style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in history.history:
            ax3.plot(history.history['lr'], linewidth=2, color='orange')
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            # If no learning rate data, show training progress
            epochs = range(1, len(history.history['accuracy']) + 1)
            ax3.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
            ax3.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
            ax3.set_title('Training Progress', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Top-K accuracy plot
        if 'top_k_categorical_accuracy' in history.history:
            ax4.plot(history.history['top_k_categorical_accuracy'], label='Training', linewidth=2)
            ax4.plot(history.history['val_top_k_categorical_accuracy'], label='Validation', linewidth=2)
            ax4.set_title('Top-K Accuracy', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Top-K Accuracy')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # If no top-k data, show loss progress
            ax4.plot(history.history['loss'], label='Training Loss', linewidth=2)
            ax4.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            ax4.set_title('Loss Progress', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("✅ Training plots saved to plots/")
        
    def save_model_data(self, model, words, classes):
        """Save model and metadata with versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f"models/chatbot_model_{timestamp}.h5"
        model.save(model_path)
        
        # Save latest model (overwrite)
        model.save("models/chatbot_model_latest.h5")
        
        # Save words and classes with versioning
        data_files = {
            f"data/words_{timestamp}.pkl": words,
            f"data/classes_{timestamp}.pkl": classes,
            "data/words_latest.pkl": words,
            "data/classes_latest.pkl": classes
        }
        
        for filepath, data in data_files.items():
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        
        # Save model metadata
        metadata = {
            'timestamp': timestamp,
            'model_path': model_path,
            'vocabulary_size': len(words),
            'num_classes': len(classes),
            'classes': classes,
            'config': self.config,
            'model_summary': []
        }
        
        # Capture model summary
        summary_lines = []
        model.summary(print_fn=summary_lines.append)
        metadata['model_summary'] = summary_lines
        
        # Save metadata
        with open(f"models/metadata_{timestamp}.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        with open("models/metadata_latest.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Model saved: {model_path}")
        self.logger.info("✅ All training artifacts saved successfully")
        
    def run_training_pipeline(self):
        """Complete training pipeline with error handling"""
        try:
            start_time = datetime.now()
            self.logger.info("🎯 Starting complete training pipeline...")
            
            # Load and validate data
            intents = self.load_intents()
            if not intents["intents"]:
                self.logger.error("❌ No valid intents found. Exiting.")
                return False
            
            # Preprocess data
            words, classes, documents = self.preprocess_data(intents)
            
            # Create training data
            train_x, train_y = self.create_training_data(words, classes, documents)
            
            # Validate training data
            if len(documents) < 20:
                self.logger.warning("⚠️  Small training dataset - consider adding more data")
            
            # Build model
            model = self.build_model(len(words), len(classes))
            
            # Train model
            model, history, (x_val, y_val) = self.train_model(model, train_x, train_y)
            
            # Evaluate model
            scores, report = self.evaluate_model(model, x_val, y_val, classes)
            
            # Create visualizations
            self.plot_training_history(history)
            
            # Save everything
            self.save_model_data(model, words, classes)
            
            # Training summary
            duration = datetime.now() - start_time
            self.logger.info("=" * 60)
            self.logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info(f"⏱️  Total training time: {duration}")
            self.logger.info(f"🎯 Final validation accuracy: {scores[1]*100:.2f}%")
            self.logger.info(f"📊 Model complexity: {model.count_params():,} parameters")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training pipeline failed: {str(e)}")
            import traceback
            self.logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return False

def main():
    """Main execution function"""
    trainer = ChatbotTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Check the following directories for outputs:")
        print("   - models/     → Trained model files")
        print("   - plots/      → Training visualizations")
        print("   - logs/       → Training logs")
        print("   - data/       → Processed data files")
    else:
        print("\n❌ Training failed. Check logs for details.")

if __name__ == "__main__":
    main()