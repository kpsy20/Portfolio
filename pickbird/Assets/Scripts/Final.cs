using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class Final : MonoBehaviour {
    public Text final;
    public Text final2;
    private int t0, s0, t1, s1;
    private int temp;
    public static List<int> list = new List<int>();
    // Use this for initialization
    void Start() {
        int temp0 = InputController.fakenumber.Count;
        //일단 주작놈 넣기
        s0 = ButtonController.sn;
        
        while(temp0>0)
        {
            if (s0 > 0)
            {
                if (InputController.fakenumber[temp0 - 1] <= ButtonController.tn)
                {
                    list.Add(InputController.fakenumber[temp0 - 1]);
                    s0 = s0 - 1;
                }
            }
            temp0--;
        }
        s1 = ButtonController.sn;
        t0 = ButtonController.tn;
        //s0이 더 넣을 갯수.

        
        while(s0>0)
        {
            //범위 안에 있는 놈들만 추림

            //범위 안에 있는 놈들 넣음
            //나머지 놈들 넣음
            bool check = true;
            while (check)
            {
                check = false;
                temp = (int)Random.Range((float)1, (float)t0+1);
                if (list.Count > 0)
                {
                    int temp2 = list.Count;
                    for (; temp2 > 0; temp2--)
                    {
                        if (list[temp2 - 1] == temp)
                        {
                            check = true;
                        }
                    }
                }
            }
            
            list.Add(temp);
            s0 = s0 - 1;
        }
        int last = list.Count;
        list.Sort();
        if (ButtonController.lang == true)
        {
            for (; last > 0; last--)
            {
                final.text = final.text + list[last - 1].ToString();
                if (last != 1) { final.text = final.text + "번, "; }
                else { final.text = final.text + "번 입니다."; }
            }
        }
        else
        {
            for (; last > 0; last--)
            {
                final.text = final.text + list[last - 1].ToString();
                if (last != 1) { final.text =  final.text + ", "; }
                else { final.text = final.text + "."; }
            }
        }
        if (ButtonController.lang == true) { final2.text = "총 인원 " + t0.ToString() + "명중 걸린 사람 " + s1.ToString() + "명은"; }
        else
        {
            final2.text = "Selected " + s1.ToString() + " of " + t0.ToString() + " people(person) in total are(is)";
        }

    }
	
	// Update is called once per frame
	void Update () {

          
        
	}
}
