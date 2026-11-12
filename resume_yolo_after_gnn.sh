#!/bin/bash
while kill -0 73976 2>/dev/null; do sleep 30; done
kill -CONT 74067 74069 74099 74100 74110 74129 74134 74135 74142 74146 74149 2>/dev/null
echo "$(date) GNN eval finished; YOLO training resumed" >> "/Users/law/cv project 3/resume_yolo_after_gnn.log"
