using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class InputController : MonoBehaviour {
    public InputField total;
    public InputField select;
    public static int totali=0;
    public static int selecti=0;
    public static List<int> fakenumber = new List<int>();
    public static bool enter = false;
   
    
    // Use this for initialization
    void Start () {
		
	}
	//48 ~ 57 ASCII
	// Update is called once per frame
	void Update () {
        int tl = total.text.Length;
        int sl = select.text.Length;

        if (tl == 0)
        {
            totali = 0;
        }
        if(sl == 0)
        {
            selecti = 0;
        }
        if (tl > 0)
        {
            try
            {
                totali = int.Parse(total.text);
            }
            catch(Exception e)
            {
                Debug.Log("다시 입력해 주세요");
                total.text = "";
            }
        }
        if (sl > 0)
        {
            try
            {
                selecti = int.Parse(select.text);
            }
            catch(Exception e)
            {
                Debug.Log("다시 입력해 주세요");
                select.text = "";
            }
        }
    }
}
