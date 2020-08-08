using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
public class ButtonController : MonoBehaviour {
    public static int tn;
    public static int sn;
    public static bool lang = true;
    public void Language()
    {
        lang = (!lang);
        //true = 한국어, false = 영어
    }
    public void yout()
    {
         Application.OpenURL("https://www.youtube.com/channel/UCn4WUi0lO5si_PDFRbBaR9w?view_as=subscriber");
    }
    public void Ads()
    {
        SceneManager.LoadScene("Ads");
    }
    public void Information()
    {
        SceneManager.LoadScene("Information");
    }
    public void Menu()
    {
        int temp = Final.list.Count;
        for (; temp > 0; temp--)
        {
            Final.list.RemoveAt(temp - 1);
        }
        SceneManager.LoadScene("Menu");
    }
    public void GOMenu()
    {
        //list초기화 주작도 초기화?
        int temp = Final.list.Count;
        for(;temp>0;temp--)
        {
            Final.list.RemoveAt(temp - 1);
        }
        SceneManager.LoadScene("Main");
    }
    public void MenuGUI()
    {
        SceneManager.LoadScene("Main");
    }
    public void FakeGUI()
    {
        SceneManager.LoadScene("Fake");
        InputController.enter = true;
    }
    public void SaveGUI()
    {
        if (FakeManager.fl == 0)
        {
            Debug.Log("다시 잘 입력해 주세요.");
        }
        if(FakeManager.fl !=0)
        {
            int temp = InputController.fakenumber.Count;
            bool temp2 = true;
            for (; temp > 0; temp--)
            {
                if (InputController.fakenumber[temp - 1] == FakeManager.fakei)
                {
                    temp2 = false;
                }
            }

            if (temp2)
            {
                InputController.fakenumber.Add(FakeManager.fakei);
                FakeManager.check = true;
            }
            else
            {
                Debug.Log("중복되었습니다. 다시 입력해 주세요.");
            }

            
        }
    }
    public void ResetGUI()
    {
        while(InputController.fakenumber.Count > 0)
        {
            InputController.fakenumber.RemoveAt(InputController.fakenumber.Count - 1);
        }
    }
    public void FakeMainGUI()
    {
        SceneManager.LoadScene("Menu");
        //주작하기 버튼 안보이게

    }

    public void GO()
    {
        if (InputController.totali > 0 && InputController.selecti >0 && InputController.totali >=InputController.selecti)
        {
            
            SceneManager.LoadScene("GO");
            tn=InputController.totali;
            sn = InputController.selecti;
        }
        else
        {
            if(InputController.totali < InputController.selecti)
            {
                Debug.Log("총 인원이 당첨 인원보다 적습니다.");
            }
            else if(InputController.totali == 0 && InputController.selecti == 0)
            {
                Debug.Log("총 인원과 당첨 인원을 입력해 주세요.");
            }
            else if(InputController.totali == 0)
            {
                Debug.Log("총 인원을 입력해 주세요.");
            }
            else if(InputController.selecti == 0)
            {
                Debug.Log("당첨 인원을 입력해 주세요.");
            }
        }
    }
}
