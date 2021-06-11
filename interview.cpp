struct Node {
  int value;
  Node *next;
};

Node *g_head;

/*

Node1 {
  value = 1;
  pointrer-.Node2
}

//
*/
// 2 -> 4
// 4 -> 2
Node *remove(int value) {

  Node *first_value = g_head;
  if (g_head->value == value) {

    Node *nextPointer = g_head->next;
    delete g_head;

    return nextPointer;
  } else {

    if (g_head->next != nullptr) {
      g_head = g_head->next;
      Node *nodeToReplace = remove(value);

      if (nodeToReplace != nullptr) {
        first_value->next = nodeToReplace;
      }
    }
    g_head = first_value;
    return nullptr;
  }
}

void remove(int value) {
  Node *start = g_head;
  Node *next;
  while (g_head->next != nullptr) {
    if (g_head->value == value) {
      next = g_head->next;
      delete g_head;
      g_head = start;

      // TODO: go to the next one or previous one and change value accordingly

      return;
    }
  }
  g_head = start;
}
