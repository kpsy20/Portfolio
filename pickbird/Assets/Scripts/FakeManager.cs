using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class FakeManager : MonoBehaviour {
    public InputField fake;
    public Text selectt;
    public static int fakei=0;
    public static int fl=0;
    public static bool check = false;

    public Text ex1, ex2, enter, reset, saved, ad, gomenu; 
	// Use this for initialization
	void Start () {
        fake.text = "";
        selectt.text = "";
	}
	
	// Update is called once per frame
	void Update () {
        fl = fake.text.Length;
        if(fl==0)
        {
            fakei = 0;
        }
        if(fl>0)
        {
            try
            {
                fakei = int.Parse(fake.text);
            }
            catch (Exception e)
            {
                Debug.Log("다시 입력해 주세요");
                fake.text = "";
            }
        }
        if(InputController.fakenumber.Count == 0)
        {
            selectt.text = "";
        }
        if(InputController.fakenumber.Count>0 && check == true)
        {
            int temp = InputController.fakenumber.Count;
            selectt.text = selectt.text + InputController.fakenumber[temp - 1]+", ";
            check = false;
        }
        if (ButtonController.lang == true)
        {
            ex1.text = "1.무조건 걸릴 번호를 입력해 주세요" + "\n" + "2.저장 버튼 누르기" + "\n\n" + "*총 인원수를 생각해서 입력해 주세요*" + "\n" + "*이 페이지는 나가면 다시 돌아오지 못합니다*";
            ex2.text = "광고를 한번 볼 때 마다" + "\n" + "조작할 수 있는 인원이" + "\n" + "한명씩 늘어납니다.";
            enter.text = "번호 입력";
            
            reset.text = "초기화";
            saved.text = "저장된 번호";
            ad.text = "광고보기";
            gomenu.text = "메인으로";
            

        }
        if(ButtonController.lang == false)
        {
            ex1.text = "1. Please enter the number to be unconditionally selected" + "\n" + "2. Press the enter button" + "\n\n" + "*Please enter in consideration of the total number of people*" + "\n" + "*If you leave this page, you will not be able to come back.*";
            ex2.text = "Each time you view an ad" + "\n" + "you increase the number of people you can manipulate.";
            enter.text = "Enter number";
            
            reset.text = "Reset";
            saved.text = "Saved number";
            ad.text = "View ad";
            gomenu.text = "Go to Menu";
           
        }
        
	}
}
