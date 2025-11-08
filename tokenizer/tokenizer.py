import copy

def merge(token_dataset, pair, idx):
  tkn_list = []
  j = 0

  while j < len(token_dataset):
      
      if j < len(token_dataset) - 1 and (token_dataset[j], token_dataset[j+1]) == pair:
          tkn_list.append(idx)
          j += 2

      else:
          tkn_list.append(token_dataset[j])
          j += 1

  if j == len(token_dataset)-1:
    tkn_list.append(token_dataset[-1])

  return tkn_list



def token_train(txt_toi:list, itos:dict, num_chars:int, new_tokens:int, sep: int):
    imax = 0
    tkn_dataset = copy.deepcopy(txt_toi)
    
    merges = {}

    while imax < new_tokens :

        couples = {}
        for c1, c2 in zip(tkn_dataset, tkn_dataset[1:]):
            if c1 != sep and c2 != sep:     # 24 should be the int corresponding to " "
              couples[(c1, c2)] = couples.get((c1, c2), 0) + 1

        max_key = max(couples.keys(), key = lambda x: couples[x])

        merges[max_key[0], max_key[1]] = imax + num_chars
        itos[imax + num_chars] = itos[max_key[0]] + itos[max_key[1]]

        idx = imax + num_chars
        tkn_dataset = merge(tkn_dataset, max_key, idx)
        imax += 1
    return tkn_dataset, merges, itos



def encode(dataset:str, merges:dict, stoi:dict, num_chars: int, new_tokens:int):
  # given a string, return list of integers (the tokens)
  dataset_ord = []

  for name in dataset:
    dataset_ord.append([stoi[c] for c in name])

  for i in range(new_tokens):

    val = num_chars + i
    pair = [k for k, v in merges.items() if v == val]
    idx = merges[pair[0]]

    for j in range(len(dataset_ord)):
      dataset_ord[j] = merge(dataset_ord[j], pair[0], idx)

  return dataset_ord



def decode(text:list, itos:dict):
  #decoding di una lista di numeri
  return "".join([itos[i] for i in text])





