from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

#main function
if __name__ == "__main__":

    model_path = "/Users/enkhdavaabatlkhagva/Code/fullstop-deep-punctuation-prediction/models/roberta-base-mn-1-task1/final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print(f"Model and tokenizer loaded from {model_path} and moved to {device}.")

    custom_text = "Гурван уулын дунд нэгэн жижигхэн тосгон байжээ тэр тосгонд нэгэн ядуу хүү амьдардаг байлаа гэвч бидний нэрийг мэддэггүй бид яах билээ"
    
    tokenizer_settings = {'return_offsets_mapping':True, 
                            'padding':False, 'truncation':True, 'stride':0, 
                            'max_length':512, 'return_overflowing_tokens':True}
    tokenized_input = tokenizer(
        custom_text,
        **tokenizer_settings
    )

    print("Tokenized input:", tokenized_input)

    input_ids = torch.tensor(tokenized_input['input_ids']).to(device)
    attention_mask = torch.tensor(tokenized_input['attention_mask']).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # The logits are the raw prediction scores. We use `argmax` to get the index of the highest score,
    # which corresponds to the predicted class ID for each token.
    predictions = torch.argmax(logits, dim=-1)


    # Move predictions back to the CPU for easier handling
    predictions = predictions.cpu().squeeze().tolist()

    # Create a mapping from ID to label
    # You'll need to define this based on your training
    id_to_label = {0: "Label_0", 1: "Label_1"}

    # Get the actual tokens from the input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    punctuated_text = []

    # Skip the first token which is the [CLS] token (or <s> for RoBERTa)
    for token, pred_id in zip(tokens[1:], predictions[1:]):
        print(f"Token: {token}, Predicted Label: pred_id", pred_id)
        # Skip padding tokens
        if token == tokenizer.pad_token or token == '</s>':
            break
        
        # RoBERTa's tokenizer often starts new words with a 'Ġ' character
        # You may need to handle this depending on your tokenizer's specifics.
        if token.startswith('Ġ'):
            punctuated_text.append(' ')
            token = token[1:] # Remove the 'Ġ'
        
        punctuated_text.append(token)
        
        # If the model predicts a punctuation, add it to the text
        if id_to_label[pred_id] == "PUNCT":
            punctuated_text.append(".") # Example: insert a period

    # Join the list of words back into a single string
    final_text = "".join(punctuated_text).strip()

    print("\n--- Final Result ---")
    print("Original Text:", custom_text)
    print("Punctuated Text:", final_text)

    