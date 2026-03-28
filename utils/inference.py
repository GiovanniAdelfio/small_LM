import re
import math
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

class ValutatoreIndovinelli:
    def __init__(self):
        print("Inizializzazione del valutatore...")
        #Per valutare la qualità dell'inglese usiamo TinyStories base
        self.giudice_id = "roneneldan/TinyStories-33M"
        self.tokenizer_giudice = AutoTokenizer.from_pretrained(self.giudice_id)
        self.modello_giudice = AutoModelForCausalLM.from_pretrained(self.giudice_id)
        self.modello_giudice.eval()
        if torch.cuda.is_available():
            self.modello_giudice = self.modello_giudice.cuda()

    def check_struttura(self, testo_generato):
        """
        Controlla se il modello ha rispettato rigorosamente la formattazione.
        Ritorna True se la struttura è perfetta, False altrimenti.
        """
        # Cerca la struttura "RIDDLE: <qualcosa> ANSWER: <qualcosa>"
        pattern = r"RIDDLE:(.*?)ANSWER:(.*)"
        match = re.search(pattern, testo_generato, re.DOTALL | re.IGNORECASE)

        if match:
            riddle = match.group(1).strip()
            answer = match.group(2).strip()
            if len(riddle) > 10 and len(answer) > 0:
                return True, riddle, answer
        return False, "", ""

    def calcola_ripetitivita(self, testo, n=3):
        """
        Calcola l'N-Gram Repetition Rate.
        Un valore vicino a 0 è ottimo (testo vario).
        Un valore vicino a 1 significa che il modello si è incantato in un loop.
        """
        parole = testo.lower().split()
        if len(parole) < n:
            return 0.0

        #Creiamo tutti i gruppi di 3 parole (trigrammi)
        ngrams = [" ".join(parole[i:i+n]) for i in range(len(parole)-n+1)]
        ngrams_unici = set(ngrams)

        # 1 - (trigrammi_unici/totale_trigrammi)
        tasso_ripetizione = 1.0 - (len(ngrams_unici) / len(ngrams))
        return round(tasso_ripetizione, 3)

    def check_coerenza_contesto(self, topic, answer_generata, riddle_generato):
        """
        Valuta se il modello è andato fuori tema.
        Controlla se la parola chiave del prompt compare nella risposta o nell'indovinello.
        """
        topic = topic.lower().strip()
        answer_generata = answer_generata.lower()
        riddle_generato = riddle_generato.lower()

        if topic in answer_generata:
            return 2
        elif topic in riddle_generato:
            return 1
        else:
            return 0

    def calcola_perplessita(self, testo):
        """
        La Perplessità (Perplexity) valuta quanto è 'naturale' e grammaticalmente corretto il testo.
        Più è BASSA, più il testo ha senso logico e sintattico.
        """
        # Puliamo il testo dai tag speciali
        testo_pulito = testo.replace("RIDDLE:", "").replace("ANSWER:", "").replace("[BOS]", "").replace("[SEP]", "").replace("[EOS]", "")

        encodings = self.tokenizer_giudice(testo_pulito, return_tensors="pt")
        input_ids = encodings.input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.no_grad():
            outputs = self.modello_giudice(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return round(perplexity.item(), 2)
