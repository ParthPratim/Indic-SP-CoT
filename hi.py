import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List

class HindiIndicNER:
    def __init__(self):
        # Load the IndicNER model and tokenizer
        model_name = "ai4bharat/IndicNER"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

    def get_entities(self, text: str) -> List[str]:
        """
        Extract named entities from Hindi text using IndicNER.
        Returns only the entity names as a list.
        
        Args:
            text (str): Input Hindi text
            
        Returns:
            List[str]: List of entity names found in the text
        """
        # Tokenize input text
        inputs = self.tokenizer(text, 
                              return_tensors="pt", 
                              padding=True, 
                              truncation=True,
                              return_offsets_mapping=True)
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
        
        # Convert predictions to entities
        entities = []
        current_entity = ""
        
        for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            label = pred.item()
            
            # Skip special tokens
            if offset[0] == offset[1]:
                continue
                
            # If it's any kind of entity (non-zero label)
            if label != 0:  # 0 is "O" (Outside)
                current_entity += text[offset[0]:offset[1]]
            else:
                if current_entity:
                    # Clean and add the entity
                    current_entity = current_entity.strip()
                    if current_entity:
                        entities.append(current_entity)
                current_entity = ""
        
        # Add final entity if exists
        if current_entity:
            current_entity = current_entity.strip()
            if current_entity:
                entities.append(current_entity)
            
        # Remove duplicates while maintaining order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
                
        return unique_entities

# Example usage
if __name__ == "__main__":
    # Initialize the NER system
    ner = HindiIndicNER()
    
    # Example Hindi text
    hindi_text = """
    नरेंद्र मोदी ने आज दिल्ली में भारतीय जनता पार्टी की बैठक में हिस्सा लिया।
    टाटा कंसल्टेंसी सर्विसेज मुंबई में स्थित है और 15 जुलाई 2023 को कंपनी ने
    नई परियोजना की घोषणा की।
    """

    hindi_text = hindi_text.encode('utf-8').decode()
    
    # Get named entities
    entities = ner.get_entities(hindi_text)
    
    # Print results
    print("\nFound entities:", entities)